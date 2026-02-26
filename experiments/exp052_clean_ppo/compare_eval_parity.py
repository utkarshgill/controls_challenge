#!/usr/bin/env python3
"""
Compare evaluation parity: tinyphysics.py (ground truth) vs tinyphysics_batched.py.

Usage example:
  python experiments/exp052_clean_ppo/compare_eval_parity.py \
    --controller exp052_clean --num-files 100 --start-idx 0 --trace-topk 3
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tinyphysics import (  # noqa: E402
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    State,
    FuturePlan,
    FUTURE_PLAN_STEPS,
)
from tinyphysics_batched import BatchedSimulator, make_ort_session  # noqa: E402


@dataclass
class FileResult:
    name: str
    tiny_cost: float
    batched_cost: float

    @property
    def diff(self) -> float:
        return self.batched_cost - self.tiny_cost

    @property
    def abs_diff(self) -> float:
        return abs(self.diff)


def pick_files(data_dir: Path, num_files: int, start_idx: int) -> List[Path]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    subset = files[start_idx : start_idx + num_files]
    if not subset:
        raise ValueError(
            f"Empty subset for start_idx={start_idx}, num_files={num_files}, total={len(files)}"
        )
    return subset


def run_tiny_for_file(
    data_file: Path, controller_name: str, model_path: Path
) -> tuple[float, np.ndarray]:
    mod = importlib.import_module(f"controllers.{controller_name}")
    controller = mod.Controller()
    model = TinyPhysicsModel(str(model_path), debug=False)
    sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
    costs = sim.rollout()
    return float(costs["total_cost"]), np.asarray(sim.current_lataccel_history, dtype=np.float64)


def run_batched_for_files(
    data_files: List[Path], controller_name: str, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    mod = importlib.import_module(f"controllers.{controller_name}")
    controllers = [mod.Controller() for _ in data_files]

    ort_session = make_ort_session(model_path)
    sim = BatchedSimulator(str(model_path), csv_files=data_files, ort_session=ort_session)
    n = len(data_files)

    def controller_fn_cpu(step_idx, target, current_lataccel, state_dict, future_plan):
        actions = np.empty(n, dtype=np.float64)
        for i, ctrl in enumerate(controllers):
            st = State(
                roll_lataccel=float(state_dict["roll_lataccel"][i]),
                v_ego=float(state_dict["v_ego"][i]),
                a_ego=float(state_dict["a_ego"][i]),
            )
            fp = FuturePlan(
                lataccel=future_plan["lataccel"][i].tolist(),
                roll_lataccel=future_plan["roll_lataccel"][i].tolist(),
                v_ego=future_plan["v_ego"][i].tolist(),
                a_ego=future_plan["a_ego"][i].tolist(),
            )
            actions[i] = ctrl.update(
                float(target[i]),
                float(current_lataccel[i]),
                st,
                future_plan=fp,
            )
        return actions

    def controller_fn_gpu(step_idx, sim_ref):
        # BatchedSimulator GPU path calls controller_fn(step_idx, sim_ref).
        dg = sim_ref.data_gpu
        t = step_idx
        end = min(t + FUTURE_PLAN_STEPS, sim_ref.T)
        target = dg["target_lataccel"][:, t].detach().cpu().numpy()
        current_lataccel = sim_ref.current_lataccel.detach().cpu().numpy()
        roll = dg["roll_lataccel"][:, t].detach().cpu().numpy()
        vego = dg["v_ego"][:, t].detach().cpu().numpy()
        aego = dg["a_ego"][:, t].detach().cpu().numpy()
        fp_lat = dg["target_lataccel"][:, t + 1 : end].detach().cpu().numpy()
        fp_roll = dg["roll_lataccel"][:, t + 1 : end].detach().cpu().numpy()
        fp_v = dg["v_ego"][:, t + 1 : end].detach().cpu().numpy()
        fp_a = dg["a_ego"][:, t + 1 : end].detach().cpu().numpy()

        actions = np.empty(n, dtype=np.float64)
        for i, ctrl in enumerate(controllers):
            st = State(
                roll_lataccel=float(roll[i]),
                v_ego=float(vego[i]),
                a_ego=float(aego[i]),
            )
            fp = FuturePlan(
                lataccel=fp_lat[i].tolist(),
                roll_lataccel=fp_roll[i].tolist(),
                v_ego=fp_v[i].tolist(),
                a_ego=fp_a[i].tolist(),
            )
            actions[i] = ctrl.update(
                float(target[i]),
                float(current_lataccel[i]),
                st,
                future_plan=fp,
            )
        return actions

    callback = controller_fn_gpu if getattr(sim, "_gpu", False) else controller_fn_cpu
    costs = sim.rollout(callback)["total_cost"].astype(np.float64)
    hist = sim.current_lataccel_history
    if hasattr(hist, "detach"):  # torch tensor (GPU/CPU)
        hist = hist.detach().cpu().numpy()
    preds = np.asarray(hist, dtype=np.float64)  # (N, T_max)
    return costs, preds


def first_diverge_idx(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    l = min(len(a), len(b))
    d = np.abs(a[:l] - b[:l])
    idx = np.where(d > tol)[0]
    return int(idx[0]) if idx.size else -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="exp052_clean", help="controllers/<name>.py")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--model-path", default=str(ROOT / "models" / "tinyphysics.onnx"))
    p.add_argument("--num-files", type=int, default=100)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--trace-topk", type=int, default=0, help="Extra per-step parity checks on top-K worst files")
    p.add_argument("--trace-tol", type=float, default=1e-6)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    files = pick_files(data_dir, args.num_files, args.start_idx)
    print(f"Comparing {len(files)} files: [{files[0].name} .. {files[-1].name}]")
    print(f"Controller={args.controller}  Model={model_path}")

    tiny_costs = []
    tiny_traces = []
    for f in files:
        c, tr = run_tiny_for_file(f, args.controller, model_path)
        tiny_costs.append(c)
        tiny_traces.append(tr)

    batched_costs, batched_preds = run_batched_for_files(files, args.controller, model_path)

    rows: List[FileResult] = []
    for i, f in enumerate(files):
        rows.append(FileResult(f.name, tiny_costs[i], float(batched_costs[i])))

    abs_diffs = np.array([r.abs_diff for r in rows], dtype=np.float64)
    signed = np.array([r.diff for r in rows], dtype=np.float64)
    print("\n=== Aggregate ===")
    print(f"tiny mean   : {np.mean(tiny_costs):.4f}")
    print(f"batched mean: {np.mean(batched_costs):.4f}")
    print(f"mean diff   : {np.mean(signed):+.4f} (batched - tiny)")
    print(f"mae diff    : {np.mean(abs_diffs):.4f}")
    print(f"max abs diff: {np.max(abs_diffs):.4f}")

    print("\n=== Worst per-file deltas (top 15 by |diff|) ===")
    for r in sorted(rows, key=lambda x: x.abs_diff, reverse=True)[:15]:
        print(
            f"{r.name}: tiny={r.tiny_cost:.4f}  batched={r.batched_cost:.4f}  "
            f"diff={r.diff:+.4f}"
        )

    if args.trace_topk > 0:
        print(f"\n=== Trace parity (top {args.trace_topk}) ===")
        worst = sorted(enumerate(rows), key=lambda x: x[1].abs_diff, reverse=True)[: args.trace_topk]
        for idx, row in worst:
            tr_tiny = tiny_traces[idx]
            tr_b = batched_preds[idx]
            l = min(len(tr_tiny), len(tr_b))
            mx = float(np.max(np.abs(tr_tiny[:l] - tr_b[:l])))
            div = first_diverge_idx(tr_tiny, tr_b, args.trace_tol)
            print(
                f"{row.name}: len_tiny={len(tr_tiny)} len_batched={len(tr_b)} "
                f"max|lataccel_diff|={mx:.8f} first_diverge_idx={div}"
            )


if __name__ == "__main__":
    main()
