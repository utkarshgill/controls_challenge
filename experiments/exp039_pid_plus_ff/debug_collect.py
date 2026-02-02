"""Debug what collect_episodes is actually doing"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train import (ActorCritic, HybridController, tinyphysics_model, 
                   prepare_future_plan, train_files)
from tinyphysics import TinyPhysicsSimulator
import random

print("Debugging collect_episodes step by step...")
print("="*60)

actor_critic = ActorCritic()

num_episodes = 2
all_future_plans = []
all_raw_actions = []
all_episode_costs = []

for ep_i in range(num_episodes):
    data_file = random.choice(train_files)
    print(f"\nEpisode {ep_i}: {Path(data_file).name}")
    
    controller = HybridController(actor_critic, deterministic=False)
    
    episode_plans = []
    episode_actions = []
    
    sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller, debug=False)
    
    while sim.step_idx < len(sim.data) - 1:
        state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
        future_plan_array = prepare_future_plan(futureplan, state)
        
        _, raw_ff = actor_critic.act(future_plan_array, deterministic=False)
        
        episode_plans.append(future_plan_array)
        episode_actions.append(raw_ff)
        
        sim.step()
    
    cost_dict = sim.compute_cost()
    episode_cost = cost_dict['total_cost']
    
    print(f"  Steps: {len(episode_plans)}")
    print(f"  Cost: {episode_cost:.2f}")
    print(f"  Reward per step: {-episode_cost / len(episode_plans):.4f}")
    
    all_future_plans.append(np.stack(episode_plans))
    all_raw_actions.append(np.array(episode_actions))
    all_episode_costs.append(episode_cost)

# Now create batch
max_len = max(len(ep) for ep in all_future_plans)
T, N = max_len, num_episodes

print(f"\nBatch shape: T={T}, N={N}")

rewards = np.zeros((T, N), dtype=np.float32)

for i in range(num_episodes):
    ep_len = len(all_future_plans[i])
    rewards[:ep_len, i] = -all_episode_costs[i] / ep_len
    print(f"\nEpisode {i}:")
    print(f"  ep_len: {ep_len}")
    print(f"  cost: {all_episode_costs[i]:.2f}")
    print(f"  reward per step: {-all_episode_costs[i] / ep_len:.4f}")
    print(f"  Non-zero rewards: {np.count_nonzero(rewards[:, i])}")
    print(f"  Mean of episode rewards: {rewards[:ep_len, i].mean():.4f}")
    print(f"  Mean of full column (with padding): {rewards[:, i].mean():.4f}")

print(f"\nFull batch stats:")
print(f"  rewards.shape: {rewards.shape}")
print(f"  Non-zero entries: {np.count_nonzero(rewards)}")
print(f"  Mean (all entries): {rewards.mean():.4f}")
print(f"  Mean (non-zero only): {rewards[rewards != 0].mean():.4f}")
print(f"  Std: {rewards.std():.4f}")
print(f"  Range: [{rewards.min():.4f}, {rewards.max():.4f}]")

