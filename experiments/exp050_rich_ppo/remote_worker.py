#!/usr/bin/env python3
"""
remote_worker.py — Standalone rollout worker for distributed PPO training.

Invoked via SSH from the main Mac:
    python remote_worker.py --ckpt .ckpt.pt --batch batch.json --out results.npz --workers 8 --mode train

Imports all logic from train.py so nothing is duplicated.
"""

import argparse, json, sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (
    ActorCritic, DeltaController, build_obs,
    compute_rewards, TinyPhysicsModel, TinyPhysicsSimulator, STATE_DIM,
)
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent


def _train_worker(args):
    torch.set_num_threads(1)
    csv, mdl, ckpt = args
    data = torch.load(ckpt, weights_only=False, map_location='cpu')
    ac = ActorCritic()
    ac.load_state_dict(data['ac'])
    ac.eval()
    ctrl = DeltaController(ac, deterministic=False)
    sim = TinyPhysicsSimulator(
        TinyPhysicsModel(mdl, debug=False), str(csv), controller=ctrl, debug=False)
    cost = sim.rollout()
    T = len(ctrl.traj)
    return (np.array([t['obs'] for t in ctrl.traj], np.float32),
            np.array([t['raw'] for t in ctrl.traj], np.float32),
            compute_rewards(ctrl.traj),
            np.array([t['val'] for t in ctrl.traj], np.float32),
            np.concatenate([np.zeros(T - 1, np.float32), [1.0]]),
            cost['total_cost'])


def _eval_worker(args):
    torch.set_num_threads(1)
    csv, mdl, ckpt = args
    data = torch.load(ckpt, weights_only=False, map_location='cpu')
    ac = ActorCritic()
    ac.load_state_dict(data['ac'])
    ac.eval()
    ctrl = DeltaController(ac, deterministic=True)
    sim = TinyPhysicsSimulator(
        TinyPhysicsModel(mdl, debug=False), str(csv), controller=ctrl, debug=False)
    return sim.rollout()['total_cost']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    args = parser.parse_args()

    mdl = str(ROOT / 'models' / 'tinyphysics.onnx')
    with open(args.batch) as f:
        csv_list = json.load(f)

    work = [(csv, mdl, args.ckpt) for csv in csv_list]

    if args.mode == 'train':
        results = process_map(_train_worker, work,
                              max_workers=args.workers, chunksize=10, disable=True)
        obs_list, raw_list, rew_list, val_list, done_list = [], [], [], [], []
        costs = []
        for obs, raw, rew, val, done, cost in results:
            obs_list.append(obs)
            raw_list.append(raw)
            rew_list.append(rew)
            val_list.append(val)
            done_list.append(done)
            costs.append(cost)

        ep_lens = np.array([len(o) for o in obs_list], dtype=np.int32)
        np.savez(args.out,
                 obs=np.concatenate(obs_list),
                 raw=np.concatenate(raw_list),
                 rew=np.concatenate(rew_list),
                 val=np.concatenate(val_list),
                 done=np.concatenate(done_list),
                 costs=np.array(costs, dtype=np.float32),
                 ep_lens=ep_lens)
    else:
        costs = process_map(_eval_worker, work,
                            max_workers=args.workers, chunksize=5, disable=True)
        np.savez(args.out, costs=np.array(costs, dtype=np.float32))

    print(f"[remote_worker] {args.mode} done: {len(csv_list)} files -> {args.out}")


if __name__ == '__main__':
    main()
