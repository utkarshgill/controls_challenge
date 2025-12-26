#!/usr/bin/env python3
"""
Test if AsyncVectorEnv is actually giving us speedup.
Compare sequential vs parallel rollout speed.
"""

import numpy as np
import time
import gymnasium as gym
from train_ppo_parallel import make_env, ActorCritic, PPO
import glob
import torch

# Test parameters
model_path = "./models/tinyphysics.onnx"
all_files = sorted(glob.glob("./data/*.csv"))
np.random.seed(42)
np.random.shuffle(all_files)
train_files = all_files[:1000]  # Smaller subset for testing

device = torch.device('cpu')
state_dim, action_dim = 56, 1
hidden_dim, trunk_layers, head_layers = 128, 1, 3

print("\n" + "="*60)
print("Testing AsyncVectorEnv Speedup")
print("="*60)

# Create policy
actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, trunk_layers, head_layers).to(device)
ppo = PPO(actor_critic, lr=1e-5, gamma=0.99, lamda=0.95, K_epochs=4, eps_clip=0.1, batch_size=2048, entropy_coef=0.0)

# Test 1: Sequential (1 env)
print("\n[1] Sequential (1 env)")
env_seq = gym.vector.AsyncVectorEnv([make_env(model_path, train_files)])
start = time.time()
states, _ = env_seq.reset()
steps = 0
while steps < 2000:
    actions = ppo(states)
    states, rewards, terminated, truncated, infos = env_seq.step(actions)
    steps += 1
seq_time = time.time() - start
env_seq.close()
print(f"  Time: {seq_time:.2f}s for 2000 steps = {2000/seq_time:.1f} steps/sec")

# Test 2: Parallel (8 envs)
print("\n[2] Parallel (8 envs)")
env_par = gym.vector.AsyncVectorEnv([make_env(model_path, train_files) for _ in range(8)])
start = time.time()
states, _ = env_par.reset()
steps = 0
while steps < 2000:  # Same total steps
    actions = ppo(states)
    states, rewards, terminated, truncated, infos = env_par.step(actions)
    steps += 8
par_time = time.time() - start
env_par.close()
print(f"  Time: {par_time:.2f}s for 2000 steps = {2000/par_time:.1f} steps/sec")

# Analysis
print("\n" + "="*60)
print("Results:")
print(f"  Sequential:  {2000/seq_time:6.1f} steps/sec")
print(f"  Parallel 8x: {2000/par_time:6.1f} steps/sec")
print(f"  Speedup:     {seq_time/par_time:.2f}x")
print(f"  Theoretical: 8.00x (if perfect parallelism)")
print("="*60)

if seq_time/par_time < 2.0:
    print("\n⚠️  WARNING: <2x speedup suggests GIL/ONNX bottleneck")
    print("   Consider switching to sequential rollouts")
elif seq_time/par_time < 5.0:
    print("\n✅ Decent speedup, but not perfect (ONNX overhead)")
else:
    print("\n✅ Good parallelism! AsyncVectorEnv is helping")

