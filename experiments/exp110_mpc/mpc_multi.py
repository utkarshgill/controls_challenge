#!/usr/bin/env python3
"""exp110 — Multi-GPU parallel MPC launcher

Splits N_ROUTES across N_GPUS, runs exp110 MPC on each GPU in parallel,
merges results into a single actions.npz.
"""

import subprocess, os, sys, time, json, torch
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent.parent
N_GPUS = int(os.getenv("N_GPUS", str(torch.cuda.device_count())))
N_ROUTES = int(os.getenv("N_ROUTES", "5000"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "100"))
MPC_K = int(os.getenv("MPC_K", "64"))
MPC_H = int(os.getenv("MPC_H", "50"))
MPC_W = int(os.getenv("MPC_W", "20"))
CEM_ITERS = int(os.getenv("CEM_ITERS", "10"))
N_BASIS_W = int(os.getenv("N_BASIS_W", "8"))
CEM_SIGMA = float(os.getenv("CEM_SIGMA", "0.1"))
CEM_ELITE = float(os.getenv("CEM_ELITE", "0.15"))
WARM_START_NPZ = os.getenv("WARM_START_NPZ", "")

SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"


def run_gpu_worker(gpu_id, route_start, route_end):
    """Run MPC on a slice of routes on a specific GPU."""
    n_routes = route_end - route_start
    save_path = SAVE_DIR / f"gpu{gpu_id}.npz"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ensure TRT libraries are found
    trt_lib = "/venv/main/lib/python3.12/site-packages/tensorrt_libs"
    env["LD_LIBRARY_PATH"] = trt_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA"] = "1"
    env.setdefault("TRT", "0")
    env["N_ROUTES"] = str(route_end)  # load first route_end routes
    env["ROUTE_START"] = str(route_start)  # skip first route_start
    env["BATCH_ROUTES"] = str(BATCH_ROUTES)
    env["MPC_K"] = str(MPC_K)
    env["MPC_H"] = str(MPC_H)
    env["MPC_W"] = str(MPC_W)
    env["CEM_ITERS"] = str(CEM_ITERS)
    env["N_BASIS_W"] = str(N_BASIS_W)
    env["CEM_SIGMA"] = str(CEM_SIGMA)
    env["CEM_ELITE"] = str(CEM_ELITE)
    env["SAVE_DIR"] = str(SAVE_DIR / f"gpu{gpu_id}")
    env["GPU_WORKER"] = "1"
    if WARM_START_NPZ:
        env["WARM_START_NPZ"] = WARM_START_NPZ

    cmd = [sys.executable, str(ROOT / "experiments" / "exp110_mpc" / "mpc_worker.py")]
    print(f"  GPU {gpu_id}: routes {route_start}-{route_end} ({n_routes} routes)")

    # Stream output with GPU prefix
    log_path = SAVE_DIR / f"gpu{gpu_id}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            tagged = f"  [GPU{gpu_id}] {line.rstrip()}"
            print(tagged, flush=True)
            logf.write(line)
        proc.wait()
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"  GPU {gpu_id} FAILED (code {proc.returncode})  ⏱{dt:.0f}s")
        return gpu_id, None, dt

    print(f"  GPU {gpu_id}: done  ⏱{dt:.0f}s")
    return gpu_id, SAVE_DIR / f"gpu{gpu_id}" / "actions.npz", dt


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"exp110 — Multi-GPU MPC: {N_ROUTES} routes on {N_GPUS} GPUs")
    print(f"  K={MPC_K} H={MPC_H} W={MPC_W} CEM_iters={CEM_ITERS} basis={N_BASIS_W}")
    print(f"  {N_ROUTES // N_GPUS} routes per GPU, BATCH_ROUTES={BATCH_ROUTES}")

    # Split routes evenly across GPUs
    routes_per_gpu = N_ROUTES // N_GPUS
    remainder = N_ROUTES % N_GPUS
    slices = []
    start = 0
    for g in range(N_GPUS):
        end = start + routes_per_gpu + (1 if g < remainder else 0)
        slices.append((g, start, end))
        start = end

    t0 = time.time()

    # Launch all GPUs in parallel
    with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
        futures = {
            executor.submit(run_gpu_worker, gpu_id, rs, re): gpu_id
            for gpu_id, rs, re in slices
        }
        results = {}
        for future in as_completed(futures):
            gpu_id, npz_path, dt = future.result()
            results[gpu_id] = npz_path

    total_dt = time.time() - t0

    # Merge all GPU results
    print(f"\nMerging results...")
    merged = {}
    all_costs = []
    for gpu_id in range(N_GPUS):
        npz_path = results.get(gpu_id)
        if npz_path is None or not Path(npz_path).exists():
            print(f"  GPU {gpu_id}: MISSING")
            continue
        data = np.load(npz_path)
        for key in data.files:
            merged[key] = data[key]
        print(f"  GPU {gpu_id}: {len(data.files)} routes loaded")

    # Save merged
    np.savez(SAVE_DIR / "actions.npz", **merged)
    print(f"\nSaved {len(merged)} routes to {SAVE_DIR / 'actions.npz'}")
    print(f"Total time: {total_dt:.0f}s ({total_dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
