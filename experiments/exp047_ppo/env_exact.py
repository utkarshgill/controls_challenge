"""
EXACT copy of TinyPhysicsSimulator logic as gym-like environment
No approximations - must match official simulator to numerical precision
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import (TinyPhysicsModel, State, FuturePlan, 
                         CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, 
                         MAX_ACC_DELTA, STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER, 
                         DEL_T, ACC_G, FUTURE_PLAN_STEPS)
import pandas as pd
from hashlib import md5


class ExactEnv:
    """EXACT replication of TinyPhysicsSimulator"""
    
    def __init__(self, model_path, data_path):
        self.sim_model = TinyPhysicsModel(model_path, debug=False)
        self.data_path = data_path
        self.data = self.get_data(data_path)
        self.reset()
    
    def get_data(self, data_path):
        """EXACT copy from TinyPhysicsSimulator.get_data"""
        df = pd.read_csv(data_path)
        processed_df = pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values
        })
        return processed_df
    
    def reset(self):
        """EXACT copy from TinyPhysicsSimulator.reset"""
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.current_lataccel = self.current_lataccel_history[-1]
        
        seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
        np.random.seed(seed)
        
        return self._get_obs()
    
    def get_state_target_futureplan(self, step_idx):
        """EXACT copy from TinyPhysicsSimulator"""
        state = self.data.iloc[step_idx]
        return (
            State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
            state['target_lataccel'],
            FuturePlan(
                lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
            )
        )
    
    def _get_obs(self):
        """Return observation"""
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        return {
            'state': state,
            'target': target,
            'current': self.current_lataccel,
            'futureplan': futureplan,
            'step': self.step_idx
        }
    
    def step(self, action):
        """EXACT copy of TinyPhysicsSimulator step logic"""
        # Get state/target/futureplan for current step
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        
        # Control step
        if self.step_idx < CONTROL_START_IDX:
            action = self.data['steer_command'].values[self.step_idx]
        action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
        self.action_history.append(action)
        
        # Sim step
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        
        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = target
        
        self.current_lataccel_history.append(self.current_lataccel)
        
        # Move to next step
        self.step_idx += 1
        done = (self.step_idx >= len(self.data))
        
        # Get next obs
        obs = self._get_obs() if not done else None
        
        # Reward (will compute cost at end of episode)
        reward = 0.0
        
        return obs, reward, done
    
    def compute_cost(self):
        """EXACT copy from TinyPhysicsSimulator.compute_cost"""
        target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
        
        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}


if __name__ == '__main__':
    # Test with PID
    class PIDController:
        def __init__(self):
            self.p = 0.195
            self.i = 0.100
            self.d = -0.053
            self.error_integral = 0
            self.prev_error = 0
        
        def act(self, obs):
            error = obs['target'] - obs['current']
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            return self.p * error + self.i * self.error_integral + self.d * error_diff
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("Testing EXACT environment with PID...")
    print("="*60)
    
    env = ExactEnv(str(model_path), str(data_path))
    controller = PIDController()
    
    obs = env.reset()
    done = False
    
    while not done:
        action = controller.act(obs)
        obs, reward, done = env.step(action)
    
    cost = env.compute_cost()
    
    # Debug info
    print(f"Debug info:")
    print(f"  Total steps: {len(env.data)}")
    print(f"  Final step_idx: {env.step_idx}")
    print(f"  Target history length: {len(env.target_lataccel_history)}")
    print(f"  Current history length: {len(env.current_lataccel_history)}")
    print(f"  Cost computed over: {CONTROL_START_IDX}:{COST_END_IDX}")
    print(f"  Samples in cost: {len(env.target_lataccel_history[CONTROL_START_IDX:COST_END_IDX])}")
    
    print(f"\nOur EXACT Environment:")
    print(f"  Lat cost:   {cost['lataccel_cost']:.4f}")
    print(f"  Jerk cost:  {cost['jerk_cost']:.4f}")
    print(f"  Total cost: {cost['total_cost']:.4f}")
    
    print(f"\nOfficial (tinyphysics.py --controller pid --data ./data/00000.csv):")
    print(f"  Lat cost:   1.293")
    print(f"  Jerk cost:  35.56")
    print(f"  Total cost: 100.2")
    
    print(f"\n{'='*60}")
    diff = abs(cost['total_cost'] - 100.2)
    if diff < 0.01:
        print(f"✓ EXACT MATCH (diff: {diff:.6f})")
    else:
        print(f"✗ MISMATCH: {diff:.4f} point difference")
