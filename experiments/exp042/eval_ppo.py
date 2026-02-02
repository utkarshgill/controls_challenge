"""
Evaluate trained PPO model (deterministic and stochastic).
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid
from bc_controller import BCController
from ppo_models import ActorCritic
from ppo_controller import PPOController


def main():
    model_path = "../../models/tinyphysics.onnx"
    data_dir = "../../data"
    bc_checkpoint = "./outputs/best_model.pt"
    ppo_checkpoint = "./outputs/best_ppo_model.pt"
    
    print("Loading models...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Test on different files (1000-1100)
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    test_files = all_files[1000:1100]
    
    print(f"Testing on {len(test_files)} files...\n")
    
    # Evaluate PID
    print("Evaluating PID...")
    pid_costs = []
    for file_path in tqdm(test_files, desc="PID"):
        pid_ctrl = pid.Controller()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=pid_ctrl, debug=False)
        pid_costs.append(sim.rollout()['total_cost'])
    
    # Evaluate BC
    print("\nEvaluating BC...")
    bc_costs = []
    for file_path in tqdm(test_files, desc="BC"):
        bc_ctrl = BCController(bc_checkpoint)
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=bc_ctrl, debug=False)
        bc_costs.append(sim.rollout()['total_cost'])
    
    # Evaluate PPO (stochastic)
    print("\nEvaluating PPO (stochastic)...")
    ac_ppo = ActorCritic(state_dim=55, action_dim=1)
    ac_ppo.load_state_dict(torch.load(ppo_checkpoint))
    current_std = ac_ppo.log_std.exp().item()
    print(f"  Current std: {current_std:.4f}")
    
    ppo_stoch_costs = []
    ppo_ctrl_stoch = PPOController(ac_ppo, steer_scale=2.0, collecting=False)
    for file_path in tqdm(test_files, desc="PPO (stoch)"):
        ppo_ctrl_stoch.reset()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=ppo_ctrl_stoch, debug=False)
        ppo_stoch_costs.append(sim.rollout()['total_cost'])
    
    # Evaluate PPO (deterministic)
    print("\nEvaluating PPO (deterministic)...")
    ac_ppo_det = ActorCritic(state_dim=55, action_dim=1)
    ac_ppo_det.load_state_dict(torch.load(ppo_checkpoint))
    ac_ppo_det.log_std.data.fill_(np.log(0.01))  # Near-zero std
    print(f"  Deterministic std: {ac_ppo_det.log_std.exp().item():.4f}")
    
    ppo_det_costs = []
    ppo_ctrl_det = PPOController(ac_ppo_det, steer_scale=2.0, collecting=False)
    for file_path in tqdm(test_files, desc="PPO (det)"):
        ppo_ctrl_det.reset()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=ppo_ctrl_det, debug=False)
        ppo_det_costs.append(sim.rollout()['total_cost'])
    
    # Convert to arrays
    pid_costs = np.array(pid_costs)
    bc_costs = np.array(bc_costs)
    ppo_stoch_costs = np.array(ppo_stoch_costs)
    ppo_det_costs = np.array(ppo_det_costs)
    
    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS: PPO vs BC vs PID")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<25} {'Mean':<12} {'Std':<12} {'Median':<12} {'95th %':<12}")
    print(f"{'-'*73}")
    print(f"{'PID':<25} {pid_costs.mean():<12.2f} {pid_costs.std():<12.2f} {np.median(pid_costs):<12.2f} {np.percentile(pid_costs, 95):<12.2f}")
    print(f"{'BC (baseline)':<25} {bc_costs.mean():<12.2f} {bc_costs.std():<12.2f} {np.median(bc_costs):<12.2f} {np.percentile(bc_costs, 95):<12.2f}")
    print(f"{'PPO (stochastic)':<25} {ppo_stoch_costs.mean():<12.2f} {ppo_stoch_costs.std():<12.2f} {np.median(ppo_stoch_costs):<12.2f} {np.percentile(ppo_stoch_costs, 95):<12.2f}")
    print(f"{'PPO (deterministic)':<25} {ppo_det_costs.mean():<12.2f} {ppo_det_costs.std():<12.2f} {np.median(ppo_det_costs):<12.2f} {np.percentile(ppo_det_costs, 95):<12.2f}")
    
    # Ratios vs PID
    print(f"\n{'='*70}")
    print("PERFORMANCE vs PID")
    print(f"{'='*70}\n")
    
    bc_ratio = bc_costs.mean() / pid_costs.mean()
    ppo_stoch_ratio = ppo_stoch_costs.mean() / pid_costs.mean()
    ppo_det_ratio = ppo_det_costs.mean() / pid_costs.mean()
    
    print(f"Mean cost ratio:")
    print(f"  BC:              {bc_ratio:.2f}x PID")
    print(f"  PPO (stoch):     {ppo_stoch_ratio:.2f}x PID")
    print(f"  PPO (det):       {ppo_det_ratio:.2f}x PID")
    
    # Did PPO improve over BC?
    print(f"\n{'='*70}")
    print("PPO IMPROVEMENT")
    print(f"{'='*70}\n")
    
    if ppo_det_costs.mean() < bc_costs.mean():
        improvement = (1 - ppo_det_ratio/bc_ratio) * 100
        print(f"✓ PPO (det) improved over BC by {improvement:.1f}%")
        print(f"  BC: {bc_costs.mean():.2f} → PPO: {ppo_det_costs.mean():.2f}")
    else:
        degradation = (ppo_det_ratio/bc_ratio - 1) * 100
        print(f"✗ PPO (det) is {degradation:.1f}% worse than BC")
        print(f"  BC: {bc_costs.mean():.2f} → PPO: {ppo_det_costs.mean():.2f}")
    
    # Exploration noise impact
    noise_cost = ppo_stoch_costs.mean() - ppo_det_costs.mean()
    print(f"\nExploration noise impact:")
    print(f"  Stochastic: {ppo_stoch_costs.mean():.2f}")
    print(f"  Deterministic: {ppo_det_costs.mean():.2f}")
    print(f"  Noise adds: {noise_cost:+.2f} cost")
    print(f"  Current std: {current_std:.4f}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
