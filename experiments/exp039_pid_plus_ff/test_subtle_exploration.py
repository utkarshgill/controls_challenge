"""Test that subtle exploration fixes the cost explosion"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from train import ActorCritic, collect_episodes, train_files

print("Testing subtle exploration...")
print("="*60)

actor_critic = ActorCritic()

print("\n1. Check initial log_std:")
log_std = actor_critic.log_std.data.item()
std = np.exp(log_std)
print(f"   log_std = {log_std:.3f}")
print(f"   std = exp({log_std:.3f}) = {std:.4f}")
print(f"   Expected: ~0.05 (subtle)")

print("\n2. Collect episodes with warmup=True (deterministic):")
future_plans, raw_actions, rewards, dones = collect_episodes(
    actor_critic, train_files, num_episodes=2, warmup=True
)
print(f"   Reward mean: {rewards.mean():.2f}")
print(f"   Expected: around -0.2 (cost ~100 / 500 steps)")

print("\n3. Collect episodes with warmup=False (exploration):")
future_plans, raw_actions, rewards, dones = collect_episodes(
    actor_critic, train_files, num_episodes=2, warmup=False
)
print(f"   Reward mean: {rewards.mean():.2f}")
print(f"   Expected: similar to warmup since std is small")

print("\n" + "="*60)
if abs(rewards.mean()) < 2.0:
    print("✅ SUCCESS! Exploration is subtle, costs are reasonable")
elif abs(rewards.mean()) > 5.0:
    print("❌ FAIL! Exploration still too large, costs exploding")
else:
    print("⚠️  Borderline - costs higher than ideal but not catastrophic")

