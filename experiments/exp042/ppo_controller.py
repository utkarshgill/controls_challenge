"""
PPO Controller - implements BaseController with experience collection.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from controllers import BaseController
from tinyphysics import LAT_ACCEL_COST_MULTIPLIER, DEL_T


class PPOController(BaseController):
    """
    Controller using ActorCritic for action selection.
    Collects experience during rollouts for PPO training.
    """
    
    def __init__(self, actor_critic, steer_scale=2.0, collecting=False):
        """
        Args:
            actor_critic: ActorCritic model
            steer_scale: Scale factor for actions ([-1,1] -> [-2,2])
            collecting: Whether to collect experience
        """
        super().__init__()
        self.actor_critic = actor_critic
        self.steer_scale = steer_scale
        self.collecting = collecting
        
        # PID-like state tracking
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        # Experience collection
        self.trajectory = []
    
    def reset(self):
        """Reset controller state for new episode"""
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.trajectory = []
    
    def compute_features(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute 55-dim feature vector (same as BC).
        
        Returns:
            numpy array of shape (55,)
        """
        # Base features (5)
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # Compute curvatures
        v_ego_squared = state.v_ego ** 2
        if v_ego_squared > 0.01:
            current_curvature = (current_lataccel - state.roll_lataccel) / v_ego_squared
            target_curvature = (target_lataccel - state.roll_lataccel) / v_ego_squared
        else:
            current_curvature = 0.0
            target_curvature = 0.0
        
        # Future plan curvatures (50)
        future_curvatures = []
        for i in range(len(future_plan.lataccel)):
            v_future_squared = future_plan.v_ego[i] ** 2
            if v_future_squared > 0.01:
                future_curv = (future_plan.lataccel[i] - future_plan.roll_lataccel[i]) / v_future_squared
            else:
                future_curv = 0.0
            future_curvatures.append(future_curv)
        
        # Pad to 50 if needed
        while len(future_curvatures) < 50:
            future_curvatures.append(0.0)
        future_curvatures = future_curvatures[:50]
        
        # Combine: [error, error_integral, error_diff, current_curv, target_curv, future_curv[50]]
        features = [error, self.error_integral, error_diff, current_curvature, target_curvature] + future_curvatures
        return np.array(features, dtype=np.float32)
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        BaseController interface: called by simulator at each step.
        
        Returns:
            steering command (float)
        """
        # Compute state features
        obs = self.compute_features(target_lataccel, current_lataccel, state, future_plan)
        
        # Get action from actor-critic
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            action_mean, action_std, value = self.actor_critic(obs_tensor.unsqueeze(0))
            
            # Sample action from distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action = dist.sample().squeeze()
            
            # Scale to steering range
            action = raw_action * self.steer_scale
            action = torch.clamp(action, -2.0, 2.0)
        
        # Store experience if collecting
        if self.collecting:
            self.trajectory.append({
                'obs': obs,
                'raw_action': raw_action.item(),
                'action': action.item(),
                'value': value.squeeze().item(),
                'target_lataccel': target_lataccel,
                'current_lataccel': current_lataccel,
            })
        
        return action.item()


def compute_rewards_from_trajectory(trajectory):
    """
    Compute per-step rewards from trajectory.
    
    Dense rewards based on lataccel tracking error and jerk.
    Reward = -(lataccel_error^2 * 50 + jerk^2)
    
    Args:
        trajectory: list of dicts with keys ['target_lataccel', 'current_lataccel']
        
    Returns:
        numpy array of rewards (T,)
    """
    rewards = []
    prev_lataccel = None
    
    for step in trajectory:
        target = step['target_lataccel']
        current = step['current_lataccel']
        
        # Lataccel tracking error (scaled like cost function)
        lataccel_error = (target - current) ** 2
        lataccel_penalty = lataccel_error * LAT_ACCEL_COST_MULTIPLIER
        
        # Jerk penalty
        if prev_lataccel is not None:
            jerk = (current - prev_lataccel) / DEL_T
            jerk_penalty = jerk ** 2
        else:
            jerk_penalty = 0.0
        
        # Combined reward (negative cost)
        reward = -(lataccel_penalty + jerk_penalty)
        rewards.append(reward)
        prev_lataccel = current
    
    return np.array(rewards, dtype=np.float32)


def test_ppo_controller():
    """Test PPOController with dummy rollout"""
    print("Testing PPOController...")
    
    from ppo_models import ActorCritic
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, State, FuturePlan
    
    # Create ActorCritic
    ac = ActorCritic(state_dim=55, action_dim=1)
    
    # Create controller with collection enabled
    controller = PPOController(ac, steer_scale=2.0, collecting=True)
    
    print(f"  Controller created")
    print(f"  Collecting: {controller.collecting}")
    
    # Load TinyPhysics and run short test
    model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
    
    # Quick test on one file
    from pathlib import Path
    data_path = Path("../../data")
    test_file = sorted(data_path.glob("*.csv"))[0]
    
    print(f"  Running rollout on: {test_file.name}")
    
    sim = TinyPhysicsSimulator(model, str(test_file), controller=controller, debug=False)
    costs = sim.rollout()
    
    print(f"\n  Rollout complete!")
    print(f"  Trajectory length: {len(controller.trajectory)}")
    print(f"  Total cost: {costs['total_cost']:.2f}")
    
    # Test reward computation
    rewards = compute_rewards_from_trajectory(controller.trajectory)
    print(f"\n  Rewards computed")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Mean reward: {rewards.mean():.2f}")
    print(f"  Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    # Check trajectory structure
    sample = controller.trajectory[100]  # Sample step
    print(f"\n  Sample trajectory step:")
    print(f"    Keys: {list(sample.keys())}")
    print(f"    obs shape: {sample['obs'].shape}")
    print(f"    action: {sample['action']:.4f}")
    print(f"    value: {sample['value']:.4f}")
    
    print("\n✓ PPOController test passed!")


if __name__ == '__main__':
    test_ppo_controller()
