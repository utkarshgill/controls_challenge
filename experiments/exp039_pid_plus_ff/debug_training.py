"""Debug what's happening during training"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train import ActorCritic, collect_episodes, train_files

print("Debugging training...")
print("="*60)

# Create fresh network
actor_critic = ActorCritic()

print("\n1. Checking network outputs on random input:")
dummy_input = torch.randn(1, 4, 50)
mean, std, value = actor_critic(dummy_input)
print(f"   Actor mean: {mean.item():.6f}")
print(f"   Actor std: {std.item():.6f}")
print(f"   Critic value: {value.item():.6f}")

print("\n2. Collecting one batch of episodes...")
future_plans, raw_actions, rewards, dones = collect_episodes(actor_critic, train_files, num_episodes=2)
print(f"   future_plans: {future_plans.shape}")
print(f"   raw_actions: {raw_actions.shape}")
print(f"   rewards: {rewards.shape}")
print(f"   Reward stats: mean={rewards.mean():.2f}, std={rewards.std():.2f}, range=[{rewards.min():.2f}, {rewards.max():.2f}]")

print("\n3. Checking critic values on collected states:")
T, N = rewards.shape
B = T * N
future_plans_flat = torch.from_numpy(future_plans).reshape(B, 4, 50).float()
_, _, values = actor_critic(future_plans_flat)
print(f"   Critic values: mean={values.mean().item():.2f}, std={values.std().item():.2f}")
print(f"   Range: [{values.min().item():.2f}, {values.max().item():.2f}]")

print("\n4. Checking actor outputs:")
means, stds, _ = actor_critic(future_plans_flat)
print(f"   Actor means: mean={means.mean().item():.6f}, std={means.std().item():.6f}")
print(f"   Actor stds: mean={stds.mean().item():.6f}, std={stds.std().item():.6f}")

print("\n" + "="*60)
print("Diagnosis:")
if values.std().item() < 1:
    print("✓ Critic values are reasonable")
elif values.std().item() > 100:
    print("❌ Critic values are HUGE - will cause loss explosion")
else:
    print("⚠️  Critic values moderately large")

if means.std().item() < 0.001:
    print("❌ Actor outputs are constant - network not learning")
elif means.std().item() < 0.01:
    print("⚠️  Actor outputs have low variance - might be stuck")
else:
    print("✓ Actor outputs vary")

