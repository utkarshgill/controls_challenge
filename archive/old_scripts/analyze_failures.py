#!/usr/bin/env python3
"""
SCIENTIFIC FAILURE ANALYSIS

Goal: Understand WHY BC/PPO fail catastrophically on some files.

Method:
1. Identify failure cases (cost > 500)
2. Compare properties of failures vs successes
3. Understand root cause
4. Propose general solution
"""

import numpy as np
import torch
import torch.nn as nn
import glob
import matplotlib.pyplot as plt

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, STEER_RANGE
from controllers import pid

# Load baseline results
data = np.load('baseline_results.npz', allow_pickle=True)
pid_costs = data['pid_costs']
bc_costs = data['bc_costs']
ppo_costs = data['ppo_costs']
test_files = data['test_files']

print("\n" + "="*60)
print("SCIENTIFIC FAILURE ANALYSIS")
print("="*60)

# Define failure threshold
FAILURE_THRESHOLD = 500

# Identify failures
pid_failures = pid_costs > FAILURE_THRESHOLD
bc_failures = bc_costs > FAILURE_THRESHOLD
ppo_failures = ppo_costs > FAILURE_THRESHOLD

print(f"\nFailure rate (cost > {FAILURE_THRESHOLD}):")
print(f"  PID:  {np.sum(pid_failures)}/100 ({100*np.mean(pid_failures):.1f}%)")
print(f"  BC:   {np.sum(bc_failures)}/100 ({100*np.mean(bc_failures):.1f}%)")
print(f"  PPO:  {np.sum(ppo_failures)}/100 ({100*np.mean(ppo_failures):.1f}%)")

# Identify files where BC/PPO fail but PID doesn't
bc_unique_failures = bc_failures & ~pid_failures
ppo_unique_failures = ppo_failures & ~pid_failures

print(f"\nUnique failures (where PID succeeded):")
print(f"  BC:   {np.sum(bc_unique_failures)}/100")
print(f"  PPO:  {np.sum(ppo_unique_failures)}/100")

if np.sum(bc_unique_failures) > 0:
    print(f"\nBC unique failure files:")
    for i in np.where(bc_unique_failures)[0]:
        print(f"  File {i}: {test_files[i]}")
        print(f"    PID cost:  {pid_costs[i]:.1f}")
        print(f"    BC cost:   {bc_costs[i]:.1f} (❌ {bc_costs[i]/pid_costs[i]:.1f}x worse)")

if np.sum(ppo_unique_failures) > 0:
    print(f"\nPPO unique failure files:")
    for i in np.where(ppo_unique_failures)[0]:
        print(f"  File {i}: {test_files[i]}")
        print(f"    PID cost:  {pid_costs[i]:.1f}")
        print(f"    PPO cost:  {ppo_costs[i]:.1f} (❌ {ppo_costs[i]/pid_costs[i]:.1f}x worse)")

# Find worst BC failure for detailed analysis
worst_bc_idx = np.argmax(bc_costs)
worst_bc_file = str(test_files[worst_bc_idx])

print("\n" + "="*60)
print("WORST BC FAILURE - DETAILED ANALYSIS")
print("="*60)
print(f"File: {worst_bc_file}")
print(f"  PID cost:  {pid_costs[worst_bc_idx]:.1f}")
print(f"  BC cost:   {bc_costs[worst_bc_idx]:.1f}")
print(f"  Ratio:     {bc_costs[worst_bc_idx]/pid_costs[worst_bc_idx]:.1f}x")

# Hypothesis testing: What makes this file special?
print("\n" + "="*60)
print("HYPOTHESIS TESTING")
print("="*60)

# Load the failure file and inspect
model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

# Architecture
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        state_dim, hidden_dim = 56, 128
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.ones(1) * np.log(0.05))
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.critic_out = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        trunk_features = self.trunk(state)
        actor_feat = self.actor_layers(trunk_features)
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        critic_feat = self.critic_layers(trunk_features)
        value = self.critic_out(critic_feat)
        return action_mean, action_std, value

OBS_SCALE = np.array([10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] + [1000.0] * 50, dtype=np.float32)

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class InstrumentedController:
    """Controller that logs internal state for analysis"""
    def __init__(self, network):
        self.network = network
        self.prev_error = 0.0
        self.error_integral = 0.0
        
        # Logging
        self.states = []
        self.actions = []
        self.errors = []
        self.error_integrals = []
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        
        # Get action
        with torch.no_grad():
            state_tensor = torch.from_numpy(obs * OBS_SCALE).float().unsqueeze(0)
            action_mean, _, _ = self.network(state_tensor)
            action = torch.tanh(action_mean) * STEER_RANGE[1]
        action_value = float(action.item())
        
        # Log
        error = target_lataccel - current_lataccel
        self.states.append(obs)
        self.actions.append(action_value)
        self.errors.append(error)
        self.error_integrals.append(self.error_integral)
        
        # Update
        self.error_integral += error
        self.prev_error = error
        
        return action_value

# Load BC network
network_bc = ActorCritic()
checkpoint = torch.load('bc_pid_checkpoint.pth', map_location='cpu', weights_only=False)
network_bc.load_state_dict(checkpoint['model_state_dict'], strict=False)
network_bc.eval()

print("\nRunning instrumented BC on failure file...")
bc_controller = InstrumentedController(network_bc)
sim_bc = TinyPhysicsSimulator(model, worst_bc_file, controller=bc_controller, debug=False)
sim_bc.rollout()

print("\nRunning instrumented PID on same file...")
class InstrumentedPID:
    def __init__(self):
        self.controller = pid.Controller()
        self.actions = []
        self.errors = []
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        action = self.controller.update(target_lataccel, current_lataccel, state, future_plan)
        error = target_lataccel - current_lataccel
        self.actions.append(action)
        self.errors.append(error)
        return action

pid_controller = InstrumentedPID()
sim_pid = TinyPhysicsSimulator(model, worst_bc_file, controller=pid_controller, debug=False)
sim_pid.rollout()

# Analysis
bc_actions = np.array(bc_controller.actions)
pid_actions = np.array(pid_controller.actions)
bc_errors = np.array(bc_controller.errors)
pid_errors = np.array(pid_controller.errors)
bc_error_integrals = np.array(bc_controller.error_integrals)

print("\n" + "="*60)
print("COMPARATIVE STATISTICS")
print("="*60)

print("\nAction statistics:")
print(f"  PID actions: mean={np.mean(pid_actions):.3f}, std={np.std(pid_actions):.3f}, max={np.max(np.abs(pid_actions)):.3f}")
print(f"  BC actions:  mean={np.mean(bc_actions):.3f}, std={np.std(bc_actions):.3f}, max={np.max(np.abs(bc_actions)):.3f}")

print("\nError statistics:")
print(f"  PID errors: mean={np.mean(np.abs(pid_errors)):.3f}, max={np.max(np.abs(pid_errors)):.3f}")
print(f"  BC errors:  mean={np.mean(np.abs(bc_errors)):.3f}, max={np.max(np.abs(bc_errors)):.3f}")

print("\nError integral (BC only):")
print(f"  Mean: {np.mean(bc_error_integrals):.1f}")
print(f"  Max:  {np.max(np.abs(bc_error_integrals)):.1f}")
print(f"  Final: {bc_error_integrals[-1]:.1f}")

# Check for divergence
if np.max(np.abs(bc_error_integrals)) > 100:
    print("\n⚠️  ERROR INTEGRAL RUNAWAY DETECTED!")
    print("   Hypothesis: No anti-windup → integral grows unbounded → dominates state → wrong actions")

# Check for action oscillation
action_diff_pid = np.diff(pid_actions)
action_diff_bc = np.diff(bc_actions)
print(f"\nAction smoothness (std of action changes):")
print(f"  PID: {np.std(action_diff_pid):.3f}")
print(f"  BC:  {np.std(action_diff_bc):.3f}")

if np.std(action_diff_bc) > 2 * np.std(action_diff_pid):
    print("\n⚠️  ACTION OSCILLATION DETECTED!")
    print("   Hypothesis: BC oscillates → jerk cost explodes")

# Look for specific failure point
bc_cost_per_step = (bc_errors**2 + (np.diff(sim_bc.current_lataccel_history)**2))
failure_step = np.argmax(bc_cost_per_step)
print(f"\nWorst step: {failure_step}/{len(bc_cost_per_step)}")
print(f"  Error at worst step: {bc_errors[failure_step]:.3f}")
print(f"  Action at worst step: {bc_actions[failure_step]:.3f}")
print(f"  Error integral at worst step: {bc_error_integrals[failure_step]:.1f}")

print("\n" + "="*60)
print("PROPOSED HYPOTHESES (ranked by likelihood):")
print("="*60)
print("\n1. ERROR INTEGRAL RUNAWAY")
print("   - BC/PPO have no anti-windup (error_integral unbounded)")
print("   - On some routes, integral grows to ±100+")
print("   - After normalization (×0.1), still ±10 → dominates other features")
print("   - Network trained on bounded integrals → OOD → bad actions")
print("   - Solution: Clamp error_integral to [-10, 10]")

print("\n2. STATE DISTRIBUTION SHIFT")
print("   - Training data from PID might not cover edge cases")
print("   - BC sees states during rollout it never saw in training")
print("   - Solution: Data augmentation or online correction")

print("\n3. COMPOUNDING ERRORS")
print("   - Small error → wrong action → bigger error → worse action → divergence")
print("   - Autoregressive simulator amplifies errors")
print("   - Solution: Add corrective term or safety bounds")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Test Hypothesis 1: Add error_integral anti-windup")
print("2. Re-evaluate BC with fixed state building")
print("3. If that works → apply to PPO")
print("4. If not → investigate Hypothesis 2 & 3")
print("="*60)

# Save detailed analysis
np.savez('failure_analysis.npz',
         bc_actions=bc_actions,
         pid_actions=pid_actions,
         bc_errors=bc_errors,
         pid_errors=pid_errors,
         bc_error_integrals=bc_error_integrals,
         failure_file=worst_bc_file)

print("\n✅ Saved: failure_analysis.npz")

