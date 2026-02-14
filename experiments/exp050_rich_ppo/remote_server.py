#!/usr/bin/env python3
"""
remote_server.py — Persistent TCP rollout server for distributed PPO training.

Start once on the remote machine:
    .venv/bin/python -u experiments/exp050_rich_ppo/remote_server.py --workers 10 --port 5555

Keeps a multiprocessing pool warm so there's zero startup overhead per epoch.
Uses the same batched rollout path as local training.
"""

import argparse, io, pickle, socket, struct, sys, time
import multiprocessing
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (
    ActorCritic, batched_rollout, _pool_init, _load_ckpt,
    _batched_train_worker, _batched_eval_worker,
)
from tinyphysics_batched import run_parallel_chunked

ROOT = Path(__file__).resolve().parent.parent.parent
MDL_PATH = ROOT / 'models' / 'tinyphysics.onnx'
CKPT_PATH = '/tmp/_remote_server_ckpt.pt'

# ── TCP helpers ──────────────────────────────────────────────────────

def _send(sock, data: bytes):
    sock.sendall(struct.pack('>I', len(data)))
    sock.sendall(data)

def _recv(sock) -> bytes:
    hdr = _recvall(sock, 4)
    if not hdr:
        return b''
    length = struct.unpack('>I', hdr)[0]
    return _recvall(sock, length)

def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

# ── main server ──────────────────────────────────────────────────────

def handle_request(pool, n_workers, payload):
    req = pickle.loads(payload)
    mode = req['mode']
    ckpt_bytes = req['ckpt']
    csv_list = req['csvs']

    # resolve relative paths
    csv_list = [str(ROOT / c) if not c.startswith('/') else c for c in csv_list]

    # write checkpoint to disk
    with open(str(CKPT_PATH), 'wb') as f:
        f.write(ckpt_bytes)

    t0 = time.time()

    if mode == 'train':
        res = run_parallel_chunked(pool, csv_list,
                                   _batched_train_worker, n_workers,
                                   extra_args=(str(CKPT_PATH),))
    else:
        res = run_parallel_chunked(pool, csv_list,
                                   _batched_eval_worker, n_workers,
                                   extra_args=(str(CKPT_PATH),))

    # serialize results to NPZ
    buf = io.BytesIO()
    if mode == 'train':
        obs_list, raw_list, rew_list, val_list, done_list, costs = [], [], [], [], [], []
        for r in res:
            obs_list.append(r[0])
            raw_list.append(r[1])
            rew_list.append(r[2])
            val_list.append(r[3])
            done_list.append(r[4])
            costs.append(r[5])
        np.savez(buf,
                 obs=np.concatenate(obs_list),
                 raw=np.concatenate(raw_list),
                 rew=np.concatenate(rew_list),
                 val=np.concatenate(val_list),
                 done=np.concatenate(done_list),
                 costs=np.array(costs, np.float32),
                 ep_lens=np.array([len(o) for o in obs_list], np.int32))
    else:
        costs = [r if isinstance(r, (int, float)) else r for r in res]
        np.savez(buf, costs=np.array(costs, np.float32))

    resp = buf.getvalue()
    elapsed = time.time() - t0
    print(f"  [{mode}] {len(csv_list)} csvs in {elapsed:.1f}s -> {len(resp)/1e6:.1f}MB",
          flush=True)
    return resp


def serve(port, n_workers):
    pool = multiprocessing.Pool(n_workers, initializer=_pool_init, initargs=(MDL_PATH,))
    print(f"[remote_server] pool={n_workers} workers (batched), port={port}", flush=True)
    print(f"[remote_server] ROOT={ROOT}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(1)
    print(f"[remote_server] listening on 0.0.0.0:{port}", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            print(f"[remote_server] connection from {addr}", flush=True)
            try:
                while True:
                    payload = _recv(conn)
                    if not payload:
                        break
                    resp = handle_request(pool, n_workers, payload)
                    _send(conn, resp)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                conn.close()
                print(f"[remote_server] {addr} disconnected", flush=True)
    except KeyboardInterrupt:
        print("\n[remote_server] shutting down...", flush=True)
    finally:
        srv.close()
        pool.terminate()
        pool.join()


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()
    serve(args.port, args.workers)
