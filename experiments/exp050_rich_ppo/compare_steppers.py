"""Compare BatchedSimulator vs TinyPhysicsSimulator (greedy physics, deterministic policy).

Runs N CSVs through both paths and reports max abs difference in current_lataccel
trajectories.  With greedy argmax ONNX and deterministic policy, the only differences
should be from float32/float64 numerical precision.
"""
import sys, os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH, COST_END_IDX,
    CONTROL_START_IDX, VOCAB_SIZE, LATACCEL_RANGE,
)
from tinyphysics_batched import BatchedSimulator, BatchedPhysicsModel
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    ActorCritic, DeltaController, build_obs_batch,
    HIST_LEN, STEER_RANGE, MAX_DELTA, DELTA_SCALE,
    LOG_STD_MIN, LOG_STD_MAX, BEST_PT,
)

N_TEST = 10


def greedy_predict_monkeypatch(self, input_data, temperature=1.0):
    """Greedy argmax instead of sampling."""
    res = self.ort_session.run(None, input_data)[0]
    return int(np.argmax(res[0, -1, :]))


def main():
    mdl_path = ROOT / 'models' / 'tinyphysics.onnx'
    csv_files = sorted((ROOT / 'data').glob('*.csv'))[:N_TEST]

    # Load policy
    ac = ActorCritic()
    ckpt = torch.load(BEST_PT, weights_only=False, map_location='cpu')
    ac.load_state_dict(ckpt['ac'])
    ac.eval()

    # ── Reference: TinyPhysicsSimulator with greedy physics ──
    print("Running reference (TinyPhysicsSimulator, greedy)...")
    mdl = TinyPhysicsModel(str(mdl_path), debug=False)
    mdl.predict = lambda input_data, temperature=1.0: greedy_predict_monkeypatch(mdl, input_data, temperature)

    ref_trajectories = []
    ref_costs = []
    for csv_f in csv_files:
        ctrl = DeltaController(ac, deterministic=True)
        sim = TinyPhysicsSimulator(mdl, str(csv_f), controller=ctrl, debug=False)
        cost = sim.rollout()
        ref_trajectories.append(np.array(sim.current_lataccel_history))
        ref_costs.append(cost['total_cost'])
        print(f"  ref {csv_f.name}: cost={cost['total_cost']:.2f}, "
              f"steps={len(sim.current_lataccel_history)}")

    # ── Batched N=1: run each CSV individually through BatchedSimulator ──
    print("\nRunning BatchedSimulator N=1 (greedy, one CSV at a time)...")

    def greedy_batched_predict(self, input_data, temperature=0.8, rngs=None):
        res = self.ort_session.run(None, input_data)[0]
        return np.argmax(res[:, -1, :], axis=1)

    orig_predict = BatchedPhysicsModel.predict
    BatchedPhysicsModel.predict = greedy_batched_predict

    n1_trajectories = []
    for csv_f in csv_files:
        sim1 = BatchedSimulator(str(mdl_path), [csv_f])
        h_act1  = np.zeros((1, HIST_LEN), np.float64)
        h_lat1  = np.zeros((1, HIST_LEN), np.float32)
        h_v1    = np.zeros((1, HIST_LEN), np.float32)
        h_a1    = np.zeros((1, HIST_LEN), np.float32)
        h_roll1 = np.zeros((1, HIST_LEN), np.float32)

        def make_ctrl(s, ha, hl, hv, haa, hr):
            def ctrl(step_idx, target, current_la, state_dict, future_plan):
                nonlocal ha, hl, hv, haa, hr
                rl = state_dict['roll_lataccel']; ve = state_dict['v_ego']; ae = state_dict['a_ego']
                c32, v32, a32, r32 = np.float32(current_la), np.float32(ve), np.float32(ae), np.float32(rl)
                if step_idx < CONTROL_START_IDX:
                    ha = np.concatenate([ha[:, 1:], np.zeros((1,1), np.float64)], 1)
                    hl = np.concatenate([hl[:, 1:], c32[:, None]], 1)
                    hv = np.concatenate([hv[:, 1:], v32[:, None]], 1)
                    haa= np.concatenate([haa[:,1:], a32[:, None]], 1)
                    hr = np.concatenate([hr[:, 1:], r32[:, None]], 1)
                    return np.zeros(1)
                obs = build_obs_batch(target, current_la, rl, ve, ae, ha, hl, hv, haa, hr, s.data, step_idx)
                with torch.inference_mode():
                    mu = ac.actor(torch.from_numpy(obs)).squeeze(-1).numpy()
                delta = np.clip(np.float64(mu) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
                action = np.clip(ha[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])
                ha = np.concatenate([ha[:, 1:], action[:, None]], 1)
                hl = np.concatenate([hl[:, 1:], c32[:, None]], 1)
                hv = np.concatenate([hv[:, 1:], v32[:, None]], 1)
                haa= np.concatenate([haa[:,1:], a32[:, None]], 1)
                hr = np.concatenate([hr[:, 1:], r32[:, None]], 1)
                return action
            return ctrl

        sim1.rollout(make_ctrl(sim1, h_act1, h_lat1, h_v1, h_a1, h_roll1))
        n1_trajectories.append(sim1.current_lataccel_history[0])  # (T,)
        print(f"  n1 {csv_f.name}: done")

    # ── Batched N=10: all CSVs together ──
    print("\nRunning BatchedSimulator N=10 (greedy, all together)...")

    sim = BatchedSimulator(str(mdl_path), csv_files)
    N = sim.N
    h_act  = np.zeros((N, HIST_LEN), np.float64)
    h_lat  = np.zeros((N, HIST_LEN), np.float32)
    h_v    = np.zeros((N, HIST_LEN), np.float32)
    h_a    = np.zeros((N, HIST_LEN), np.float32)
    h_roll = np.zeros((N, HIST_LEN), np.float32)

    def controller_fn(step_idx, target, current_la, state_dict, future_plan):
        nonlocal h_act, h_lat, h_v, h_a, h_roll
        roll_la = state_dict['roll_lataccel']; v_ego = state_dict['v_ego']; a_ego = state_dict['a_ego']
        cla32, ve32, ae32, rl32 = np.float32(current_la), np.float32(v_ego), np.float32(a_ego), np.float32(roll_la)
        if step_idx < CONTROL_START_IDX:
            h_act  = np.concatenate([h_act[:, 1:],  np.zeros((N, 1), np.float64)], axis=1)
            h_lat  = np.concatenate([h_lat[:, 1:],  cla32[:, None]], axis=1)
            h_v    = np.concatenate([h_v[:, 1:],    ve32[:, None]], axis=1)
            h_a    = np.concatenate([h_a[:, 1:],    ae32[:, None]], axis=1)
            h_roll = np.concatenate([h_roll[:, 1:], rl32[:, None]], axis=1)
            return np.zeros(N)
        obs = build_obs_batch(target, current_la, roll_la, v_ego, a_ego,
                              h_act, h_lat, h_v, h_a, h_roll, sim.data, step_idx)
        with torch.inference_mode():
            mu = ac.actor(torch.from_numpy(obs)).squeeze(-1).numpy()
        delta  = np.clip(np.float64(mu) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
        action = np.clip(h_act[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])
        h_act  = np.concatenate([h_act[:, 1:],  action[:, None]], axis=1)
        h_lat  = np.concatenate([h_lat[:, 1:],  cla32[:, None]], axis=1)
        h_v    = np.concatenate([h_v[:, 1:],    ve32[:, None]], axis=1)
        h_a    = np.concatenate([h_a[:, 1:],    ae32[:, None]], axis=1)
        h_roll = np.concatenate([h_roll[:, 1:], rl32[:, None]], axis=1)
        return action

    cost_dict = sim.rollout(controller_fn)
    BatchedPhysicsModel.predict = orig_predict

    batched_la = sim.current_lataccel_history  # (N, T_total)

    # ── Compare ──
    N = len(csv_files)

    # N=1 vs reference (isolates: is our code logic identical?)
    print(f"\n{'='*60}")
    print(f"N=1 (batched, one-at-a-time) vs Reference")
    print(f"{'='*60}")
    n1_max = 0.0
    for i in range(N):
        ref = ref_trajectories[i]
        n1  = n1_trajectories[i]
        L   = min(len(ref), len(n1))
        d   = np.abs(ref[:L] - n1[:L])
        mx  = np.max(d)
        n1_max = max(n1_max, mx)
        div = np.argmax(d > 1e-6) if np.any(d > 1e-6) else -1
        print(f"  {csv_files[i].name}: max_diff={mx:.12f}  first_diverge={div}")
    print(f"Overall: {n1_max:.12f}")

    # N=10 vs reference (adds: does batch size matter?)
    print(f"\n{'='*60}")
    print(f"N=10 (batched, all together) vs Reference")
    print(f"{'='*60}")
    n10_max = 0.0
    for i in range(N):
        ref = ref_trajectories[i]
        L   = min(len(ref), batched_la.shape[1])
        d   = np.abs(ref[:L] - batched_la[i, :L])
        mx  = np.max(d)
        n10_max = max(n10_max, mx)
        div = np.argmax(d > 1e-6) if np.any(d > 1e-6) else -1
        print(f"  {csv_files[i].name}: max_diff={mx:.12f}  first_diverge={div}")
    print(f"Overall: {n10_max:.12f}")

    # N=1 vs N=10 (isolates: is batching itself the issue?)
    print(f"\n{'='*60}")
    print(f"N=1 vs N=10 (pure batch-size effect)")
    print(f"{'='*60}")
    batch_max = 0.0
    for i in range(N):
        n1  = n1_trajectories[i]
        L   = min(len(n1), batched_la.shape[1])
        d   = np.abs(n1[:L] - batched_la[i, :L])
        mx  = np.max(d)
        batch_max = max(batch_max, mx)
        div = np.argmax(d > 1e-6) if np.any(d > 1e-6) else -1
        print(f"  {csv_files[i].name}: max_diff={mx:.12f}  first_diverge={div}")
    print(f"Overall: {batch_max:.12f}")

    if n1_max == 0.0:
        print("\n✓ N=1 batched matches reference EXACTLY — our code is correct.")
        if n10_max > 0.0:
            print("  N=10 diffs are purely from ONNX batch-size floating-point non-determinism.")
    elif n1_max < 1e-6:
        print("\n✓ N=1 vs ref: negligible — code is correct.")
    else:
        print(f"\n✗ N=1 vs ref differs by {n1_max:.12f} — code logic mismatch.")

    # ── Stochastic parity (sampling, per-episode seeded RNG) ──
    print(f"\n{'='*60}")
    print("STOCHASTIC PARITY (seeded sampling, N=1 vs Reference)")
    print(f"{'='*60}")

    # Reference: uses np.random.seed(md5(path) % 10**4) inside reset()
    # so each rollout gets its own seed via TinyPhysicsSimulator.__init__
    mdl_stoch = TinyPhysicsModel(str(mdl_path), debug=False)  # normal (sampling) predict
    ref_stoch = []
    for csv_f in csv_files:
        ctrl = DeltaController(ac, deterministic=True)
        sim_r = TinyPhysicsSimulator(mdl_stoch, str(csv_f), controller=ctrl, debug=False)
        sim_r.rollout()
        ref_stoch.append(np.array(sim_r.current_lataccel_history))
        print(f"  ref {csv_f.name}: done")

    # Batched N=1: uses per-episode RandomState seeded the same way
    n1_stoch = []
    for csv_f in csv_files:
        sim1 = BatchedSimulator(str(mdl_path), [csv_f])
        h_act1  = np.zeros((1, HIST_LEN), np.float64)
        h_lat1  = np.zeros((1, HIST_LEN), np.float32)
        h_v1    = np.zeros((1, HIST_LEN), np.float32)
        h_a1    = np.zeros((1, HIST_LEN), np.float32)
        h_roll1 = np.zeros((1, HIST_LEN), np.float32)

        def make_ctrl_s(s, ha, hl, hv, haa, hr):
            def ctrl(step_idx, target, current_la, state_dict, future_plan):
                nonlocal ha, hl, hv, haa, hr
                rl = state_dict['roll_lataccel']; ve = state_dict['v_ego']; ae = state_dict['a_ego']
                c32, v32, a32, r32 = np.float32(current_la), np.float32(ve), np.float32(ae), np.float32(rl)
                if step_idx < CONTROL_START_IDX:
                    ha = np.concatenate([ha[:, 1:], np.zeros((1,1), np.float64)], 1)
                    hl = np.concatenate([hl[:, 1:], c32[:, None]], 1)
                    hv = np.concatenate([hv[:, 1:], v32[:, None]], 1)
                    haa= np.concatenate([haa[:,1:], a32[:, None]], 1)
                    hr = np.concatenate([hr[:, 1:], r32[:, None]], 1)
                    return np.zeros(1)
                obs = build_obs_batch(target, current_la, rl, ve, ae, ha, hl, hv, haa, hr, s.data, step_idx)
                with torch.inference_mode():
                    mu = ac.actor(torch.from_numpy(obs)).squeeze(-1).numpy()
                delta = np.clip(np.float64(mu) * DELTA_SCALE, -MAX_DELTA, MAX_DELTA)
                action = np.clip(ha[:, -1] + delta, STEER_RANGE[0], STEER_RANGE[1])
                ha = np.concatenate([ha[:, 1:], action[:, None]], 1)
                hl = np.concatenate([hl[:, 1:], c32[:, None]], 1)
                hv = np.concatenate([hv[:, 1:], v32[:, None]], 1)
                haa= np.concatenate([haa[:,1:], a32[:, None]], 1)
                hr = np.concatenate([hr[:, 1:], r32[:, None]], 1)
                return action
            return ctrl

        sim1.rollout(make_ctrl_s(sim1, h_act1, h_lat1, h_v1, h_a1, h_roll1))
        n1_stoch.append(sim1.current_lataccel_history[0])
        print(f"  n1 {csv_f.name}: done")

    stoch_max = 0.0
    for i in range(len(csv_files)):
        ref = ref_stoch[i]
        n1  = n1_stoch[i]
        L   = min(len(ref), len(n1))
        d   = np.abs(ref[:L] - n1[:L])
        mx  = np.max(d)
        stoch_max = max(stoch_max, mx)
        div = np.argmax(d > 1e-6) if np.any(d > 1e-6) else -1
        print(f"  {csv_files[i].name}: max_diff={mx:.12f}  first_diverge={div}")
    print(f"Overall: {stoch_max:.12f}")
    if stoch_max == 0.0:
        print("\n✓ Stochastic N=1 matches reference EXACTLY — seeded RNG parity confirmed.")
    elif stoch_max < 1e-6:
        print("\n✓ Stochastic parity: negligible diffs.")
    else:
        print(f"\n✗ Stochastic parity broken: max_diff={stoch_max:.12f}")


if __name__ == '__main__':
    main()
