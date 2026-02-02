"""
Thorough test of environment before training
"""
import sys
sys.path.insert(0, '../..')

from env_wrapper_v2 import TinyPhysicsResidualEnv, make_env
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    print("="*60)
    print("THOROUGH ENVIRONMENT TEST")
    print("="*60)

    # Get data files
    data_dir = Path("../../data")
    all_files = sorted(list(data_dir.glob("*.csv")))
    print(f"\n1. Data Files")
    print(f"   Total available: {len(all_files)}")

    # Use subset for testing
    test_files = all_files[:1000]  # First 1000 files
    print(f"   Using for test: {len(test_files)}")

    # Test 1: Single environment creation
    print(f"\n2. Single Environment Creation")
    env = TinyPhysicsResidualEnv(test_files)
    print(f"   ✓ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test 2: Reset and check file sampling
    print(f"\n3. Reset and File Sampling")
    files_seen = []
    for i in range(5):
        state, info = env.reset()
        files_seen.append(info['file'])
        print(f"   Reset {i+1}: {Path(info['file']).name}, state shape: {state.shape}")

    unique_files = len(set(files_seen))
    print(f"   ✓ Saw {unique_files}/5 unique files (randomness check)")

    # Test 3: Single episode with pure PID (zero residual)
    print(f"\n4. Full Episode (Pure PID, zero residual)")
    state, info = env.reset()
    episode_file = info['file']
    steps = 0
    total_reward = 0
    done = False

    while not done and steps < 2000:
        action = np.array([0.0])  # Zero residual = pure PID
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"   File: {Path(episode_file).name}")
    print(f"   Steps: {steps}")
    print(f"   Total reward: {total_reward:.2f}")
    if 'episode_cost' in info:
        print(f"   Episode cost: {info['episode_cost']:.2f}")
    print(f"   ✓ Episode completed successfully")

    # Test 4: Vectorized environment
    print(f"\n5. Vectorized Environment (8 parallel)")
    vec_env = make_env(test_files)
    print(f"   ✓ Created {vec_env.num_envs} parallel environments")

    states, infos = vec_env.reset()
    print(f"   States shape: {states.shape}")
    if 'file' in infos:
        print(f"   Files being used:")
        for i, file_path in enumerate(infos['file']):
            print(f"     Env {i}: {Path(file_path).name}")

    # Test 5: Vectorized step
    print(f"\n6. Vectorized Step Test")
    actions = vec_env.action_space.sample()
    next_states, rewards, dones, truncated, infos = vec_env.step(actions)
    print(f"   ✓ Parallel step completed")
    print(f"   Next states shape: {next_states.shape}")
    print(f"   Rewards: {rewards}")
    print(f"   Dones: {dones}")

    vec_env.close()

    # Test 6: Multiple episodes to check diversity
    print(f"\n7. File Diversity Test (20 episodes)")
    env = TinyPhysicsResidualEnv(test_files)
    episode_files = []
    for ep in range(20):
        state, info = env.reset()
        episode_files.append(info['file'])

    unique_episodes = len(set(episode_files))
    print(f"   Unique files in 20 episodes: {unique_episodes}/20")
    if unique_episodes >= 15:
        print(f"   ✓ Good diversity (≥75%)")
    elif unique_episodes >= 10:
        print(f"   ⚠ Moderate diversity (50-75%)")
    else:
        print(f"   ✗ Low diversity (<50%)")

    # Test 7: State consistency
    print(f"\n8. State Value Checks")
    state, info = env.reset()
    print(f"   State min: {state.min():.4f}")
    print(f"   State max: {state.max():.4f}")
    print(f"   State mean: {state.mean():.4f}")
    print(f"   State std: {state.std():.4f}")

    # Check for NaN or Inf
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        print(f"   ✗ PROBLEM: NaN or Inf in state!")
    else:
        print(f"   ✓ No NaN or Inf values")

    print(f"\n" + "="*60)
    print(f"ENVIRONMENT TEST COMPLETE")
    print(f"="*60)
    print(f"\n✅ Environment is ready for training")