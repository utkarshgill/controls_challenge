#!/usr/bin/env python3
"""Multi-GPU launcher for gradient-based action optimization.

Splits routes across GPUs, runs grad_opt.py on each, merges results.
"""

import subprocess, os, sys, time, torch
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent.parent
N_GPUS = int(os.getenv("N_GPUS", str(torch.cuda.device_count())))
N_ROUTES = int(os.getenv("N_ROUTES", "5000"))
BATCH_ROUTES = int(os.getenv("BATCH_ROUTES", "50"))
GRAD_STEPS = int(os.getenv("GRAD_STEPS", "200"))
GRAD_LR = float(os.getenv("GRAD_LR", "0.01"))
WARM_START_NPZ = os.getenv("WARM_START_NPZ", "")

SAVE_DIR = Path(__file__).resolve().parent / "checkpoints" / "grad"


def run_gpu_worker(gpu_id, route_start, route_end):
    n_routes = route_end - route_start
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Libraries
    for libdir in [
        "/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib",
        "/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/venv/main/lib/python3.12/site-packages/tensorrt_libs",
    ]:
        env["LD_LIBRARY_PATH"] = libdir + ":" + env.get("LD_LIBRARY_PATH", "")
    env["N_ROUTES"] = str(route_end)
    env["ROUTE_START"] = str(route_start)
    env["BATCH_ROUTES"] = str(BATCH_ROUTES)
    env["GRAD_STEPS"] = str(GRAD_STEPS)
    env["GRAD_LR"] = str(GRAD_LR)
    env["SAVE_DIR"] = str(SAVE_DIR / f"gpu{gpu_id}")
    if WARM_START_NPZ:
        env["WARM_START_NPZ"] = WARM_START_NPZ

    cmd = [sys.executable, str(ROOT / "experiments" / "exp110_mpc" / "grad_opt.py")]
    print(f"  GPU {gpu_id}: routes {route_start}-{route_end} ({n_routes} routes)")

    log_path = SAVE_DIR / f"gpu{gpu_id}.log"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / f"gpu{gpu_id}").mkdir(parents=True, exist_ok=True)

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

    print(f"Gradient optimization: {N_ROUTES} routes on {N_GPUS} GPUs")
    print(f"  batch={BATCH_ROUTES}, steps={GRAD_STEPS}, lr={GRAD_LR}")

    routes_per_gpu = N_ROUTES // N_GPUS
    remainder = N_ROUTES % N_GPUS
    slices = []
    start = 0
    for g in range(N_GPUS):
        end = start + routes_per_gpu + (1 if g < remainder else 0)
        slices.append((g, start, end))
        start = end

    t0 = time.time()

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

    # Merge
    print(f"\nMerging results...")
    merged = {}
    for gpu_id in range(N_GPUS):
        npz_path = results.get(gpu_id)
        if npz_path is None or not Path(npz_path).exists():
            print(f"  GPU {gpu_id}: MISSING")
            continue
        data = np.load(npz_path)
        for key in data.files:
            merged[key] = data[key]
        print(f"  GPU {gpu_id}: {len(data.files)} routes loaded")

    np.savez(SAVE_DIR / "actions.npz", **merged)
    print(f"\nSaved {len(merged)} routes to {SAVE_DIR / 'actions.npz'}")
    print(f"Total time: {total_dt:.0f}s ({total_dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
