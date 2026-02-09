#!/usr/bin/env python3
"""
Comprehensive Data Analysis for comma.ai Controls Challenge
============================================================
"Become one with the data" - Andrej Karpathy

This script analyzes ALL 20,000 trajectory files to understand:
1. What are we actually trying to control?
2. What are the input/output distributions?
3. Why is this problem hard?
4. What patterns exist in the data?
5. What is the "texture" of the data?

Run: python analysis/data_deep_dive.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants from tinyphysics.py
ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100  # First 100 steps are warmup (10 seconds)
COST_END_IDX = 500       # Last 100 steps ignored in cost
CONTEXT_LENGTH = 20
DEL_T = 0.1  # 100ms timestep
LAT_ACCEL_COST_MULTIPLIER = 50.0
FUTURE_PLAN_STEPS = 50   # 5 seconds of future

DATA_PATH = Path(__file__).parent.parent / "data"

print("=" * 80)
print("COMMA.AI CONTROLS CHALLENGE - COMPREHENSIVE DATA ANALYSIS")
print("=" * 80)
print()

# =============================================================================
# PART 1: LOAD ALL DATA
# =============================================================================
print("PART 1: LOADING ALL 20,000 TRAJECTORIES")
print("-" * 40)

files = sorted(DATA_PATH.glob("*.csv"))
print(f"Found {len(files)} trajectory files")

# Load everything into memory for fast analysis
all_data = []
file_stats = []

for f in tqdm(files, desc="Loading trajectories"):
    df = pd.read_csv(f)
    
    # Compute derived features (matching tinyphysics.py)
    df['roll_lataccel'] = np.sin(df['roll'].values) * ACC_G
    df['steer_command_flipped'] = -df['steerCommand']  # Match simulator convention
    
    # Compute per-file statistics
    stats = {
        'file': f.name,
        'n_rows': len(df),
        'v_ego_mean': df['vEgo'].mean(),
        'v_ego_std': df['vEgo'].std(),
        'v_ego_min': df['vEgo'].min(),
        'v_ego_max': df['vEgo'].max(),
        'a_ego_mean': df['aEgo'].mean(),
        'a_ego_std': df['aEgo'].std(),
        'target_lataccel_mean': df['targetLateralAcceleration'].mean(),
        'target_lataccel_std': df['targetLateralAcceleration'].std(),
        'target_lataccel_min': df['targetLateralAcceleration'].min(),
        'target_lataccel_max': df['targetLateralAcceleration'].max(),
        'target_lataccel_range': df['targetLateralAcceleration'].max() - df['targetLateralAcceleration'].min(),
        'steer_cmd_mean': df['steer_command_flipped'].mean(),
        'steer_cmd_std': df['steer_command_flipped'].std(),
        'roll_lataccel_mean': df['roll_lataccel'].mean(),
        'roll_lataccel_std': df['roll_lataccel'].std(),
        # Dynamics
        'target_jerk_mean': np.abs(np.diff(df['targetLateralAcceleration']) / DEL_T).mean(),
        'target_jerk_max': np.abs(np.diff(df['targetLateralAcceleration']) / DEL_T).max(),
        'steer_jerk_mean': np.abs(np.diff(df['steer_command_flipped']) / DEL_T).mean(),
    }
    file_stats.append(stats)
    all_data.append(df)

stats_df = pd.DataFrame(file_stats)
print(f"\nLoaded {len(all_data)} trajectories, each with {all_data[0].shape[0]} timesteps")
print(f"Total data points: {len(all_data) * all_data[0].shape[0]:,} = 12 million!")
print()

# =============================================================================
# PART 2: GLOBAL DISTRIBUTIONS
# =============================================================================
print("PART 2: UNDERSTANDING GLOBAL DISTRIBUTIONS")
print("-" * 40)

# Combine all data for global analysis
combined = pd.concat(all_data, ignore_index=True)

print("\n📊 COLUMN STATISTICS (all 20k files, all 600 timesteps):")
print("-" * 60)
for col in ['vEgo', 'aEgo', 'roll', 'targetLateralAcceleration', 'steer_command_flipped', 'roll_lataccel']:
    data = combined[col]
    print(f"\n{col}:")
    print(f"  Mean:   {data.mean():>10.4f}")
    print(f"  Std:    {data.std():>10.4f}")
    print(f"  Min:    {data.min():>10.4f}")
    print(f"  Max:    {data.max():>10.4f}")
    print(f"  5%:     {data.quantile(0.05):>10.4f}")
    print(f"  50%:    {data.quantile(0.50):>10.4f}")
    print(f"  95%:    {data.quantile(0.95):>10.4f}")

# =============================================================================
# PART 3: WHAT MAKES THIS PROBLEM HARD?
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: WHY IS THIS PROBLEM HARD?")
print("=" * 80)

# 3.1: Target lateral acceleration analysis
print("\n🎯 TARGET LATERAL ACCELERATION ANALYSIS:")
print("-" * 40)
target = combined['targetLateralAcceleration']
print(f"Range: [{target.min():.3f}, {target.max():.3f}] m/s²")
print(f"This is [{target.min()/ACC_G:.3f}, {target.max()/ACC_G:.3f}] g")
print(f"\n95% of values between: [{target.quantile(0.025):.3f}, {target.quantile(0.975):.3f}]")

# How much does target change per timestep?
target_changes = []
for df in all_data:
    target_changes.extend(np.diff(df['targetLateralAcceleration']))
target_changes = np.array(target_changes)

print(f"\nTarget change per 100ms timestep:")
print(f"  Mean abs change: {np.abs(target_changes).mean():.4f} m/s²")
print(f"  Max abs change:  {np.abs(target_changes).max():.4f} m/s²")
print(f"  Std of change:   {target_changes.std():.4f} m/s²")

# 3.2: Relationship between steer command and target
print("\n🎮 STEER COMMAND vs TARGET RELATIONSHIP:")
print("-" * 40)
steer = combined['steer_command_flipped']
corr = combined['targetLateralAcceleration'].corr(steer)
print(f"Correlation: {corr:.4f}")
print("  → Steer command is negatively correlated with target!")
print("  → This is counterintuitive - why?")

# 3.3: The role of vehicle state
print("\n🚗 VEHICLE STATE INFLUENCE:")
print("-" * 40)
v_ego = combined['vEgo']
print(f"Speed range: [{v_ego.min():.1f}, {v_ego.max():.1f}] m/s")
print(f"           = [{v_ego.min()*3.6:.1f}, {v_ego.max()*3.6:.1f}] km/h")
print(f"           = [{v_ego.min()*2.237:.1f}, {v_ego.max()*2.237:.1f}] mph")

# How does speed affect the relationship?
print("\nCorrelation by speed bucket:")
speed_buckets = [(0, 10), (10, 20), (20, 30), (30, 40)]
for lo, hi in speed_buckets:
    mask = (combined['vEgo'] >= lo) & (combined['vEgo'] < hi)
    if mask.sum() > 100:
        corr = combined.loc[mask, 'targetLateralAcceleration'].corr(
            combined.loc[mask, 'steer_command_flipped']
        )
        print(f"  Speed {lo:>2}-{hi:<2} m/s: corr = {corr:>7.4f} (n={mask.sum():>8,})")

# 3.4: The roll factor
print("\n⛰️ ROAD ROLL INFLUENCE:")
print("-" * 40)
roll = combined['roll']
roll_lataccel = combined['roll_lataccel']
print(f"Roll angle range: [{roll.min():.4f}, {roll.max():.4f}] rad")
print(f"               = [{np.degrees(roll.min()):.2f}, {np.degrees(roll.max()):.2f}] deg")
print(f"Roll lat accel: [{roll_lataccel.min():.3f}, {roll_lataccel.max():.3f}] m/s²")
print(f"\nRoll accounts for up to {roll_lataccel.abs().max():.2f} m/s² of lateral acceleration!")
print("This is a DISTURBANCE the controller must compensate for.")

# 3.5: The delay/dynamics problem
print("\n⏱️ CONTROL DELAY & DYNAMICS:")
print("-" * 40)
print(f"Control starts at step {CONTROL_START_IDX} (t={CONTROL_START_IDX*DEL_T}s)")
print(f"Cost computed from step {CONTROL_START_IDX} to {COST_END_IDX}")
print(f"Future plan: {FUTURE_PLAN_STEPS} steps = {FUTURE_PLAN_STEPS*DEL_T}s lookahead")
print(f"\nThe ONNX physics model has:")
print(f"  - Context length: {CONTEXT_LENGTH} timesteps = {CONTEXT_LENGTH*DEL_T}s")
print(f"  - Tokenized lataccel history → transformer-like prediction")
print(f"  - Stochastic output (temperature=0.8)")
print("\n⚠️ KEY INSIGHT: The physics model is a BLACK BOX neural network!")
print("   We don't know the true dynamics - only learned approximations.")

# =============================================================================
# PART 4: TRAJECTORY DIVERSITY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: TRAJECTORY DIVERSITY")
print("=" * 80)

print("\n📈 PER-TRAJECTORY STATISTICS:")
print("-" * 40)
print(f"Target lataccel range per trajectory:")
print(f"  Mean range: {stats_df['target_lataccel_range'].mean():.3f} m/s²")
print(f"  Std range:  {stats_df['target_lataccel_range'].std():.3f} m/s²")
print(f"  Min range:  {stats_df['target_lataccel_range'].min():.3f} m/s²")
print(f"  Max range:  {stats_df['target_lataccel_range'].max():.3f} m/s²")

print(f"\nSpeed variation per trajectory:")
print(f"  Mean std: {stats_df['v_ego_std'].mean():.3f} m/s")
print(f"  Max std:  {stats_df['v_ego_std'].max():.3f} m/s")

# Find trajectories with extreme characteristics
print("\n🔥 EXTREME TRAJECTORIES:")
print("-" * 40)

# Hardest trajectories (highest target variance)
hardest = stats_df.nlargest(10, 'target_lataccel_std')
print("\nTop 10 hardest (highest target variance):")
for _, row in hardest.iterrows():
    print(f"  {row['file']}: target_std={row['target_lataccel_std']:.3f}, "
          f"range=[{row['target_lataccel_min']:.2f}, {row['target_lataccel_max']:.2f}]")

# Easiest trajectories
easiest = stats_df.nsmallest(10, 'target_lataccel_std')
print("\nTop 10 easiest (lowest target variance):")
for _, row in easiest.iterrows():
    print(f"  {row['file']}: target_std={row['target_lataccel_std']:.3f}, "
          f"range=[{row['target_lataccel_min']:.2f}, {row['target_lataccel_max']:.2f}]")

# =============================================================================
# PART 5: FREQUENCY DOMAIN ANALYSIS (Simple FFT without scipy)
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: FREQUENCY CONTENT OF SIGNALS")
print("=" * 80)

print("\nAnalyzing frequency content of target lateral acceleration...")

# Combine all target signals for aggregate frequency analysis (first 1000 trajectories)
all_targets = np.concatenate([df['targetLateralAcceleration'].values for df in all_data[:1000]])

# Simple FFT analysis using numpy
n = len(all_targets)
fft_vals = np.fft.fft(all_targets)
freqs = np.fft.fftfreq(n, d=DEL_T)  # DEL_T = 0.1s = 100ms
psd = np.abs(fft_vals)**2 / n

# Only positive frequencies
pos_mask = freqs > 0
freqs = freqs[pos_mask]
psd = psd[pos_mask]

print(f"\nFrequency analysis (first 1000 trajectories, {n:,} samples):")
print(f"  Sampling rate: {FPS} Hz")
print(f"  Nyquist freq: {FPS/2} Hz")
print(f"  Dominant frequencies:")
peak_idxs = np.argsort(psd)[-5:][::-1]
for idx in peak_idxs:
    print(f"    {freqs[idx]:.3f} Hz: power={psd[idx]:.4f}")

# =============================================================================
# PART 6: CONTROL CHALLENGE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: THE CONTROL CHALLENGE")
print("=" * 80)

# What a perfect controller needs to do
print("\n📐 COST FUNCTION BREAKDOWN:")
print("-" * 40)
print("Total Cost = lataccel_cost × 50 + jerk_cost")
print()
print("lataccel_cost = mean((target - actual)²) × 100")
print("jerk_cost = mean((Δactual / 0.1s)²) × 100")
print()
print("→ Tracking error is weighted 50× more than jerk!")
print("→ But jerk still matters - smooth control required")

# Analyze the baseline (logged steer commands)
print("\n📊 BASELINE ANALYSIS (Logged Expert Commands):")
print("-" * 40)

# Compute what the logged commands' jerk would be
all_steer_jerks = []
for df in all_data:
    steer = df['steer_command_flipped'].values
    steer_jerk = np.diff(steer) / DEL_T
    all_steer_jerks.extend(steer_jerk[CONTROL_START_IDX-1:COST_END_IDX-1])

all_steer_jerks = np.array(all_steer_jerks)
print(f"Logged steering command jerk (in control window):")
print(f"  Mean abs: {np.abs(all_steer_jerks).mean():.4f} /s")
print(f"  Std:      {all_steer_jerks.std():.4f} /s")
print(f"  Max abs:  {np.abs(all_steer_jerks).max():.4f} /s")

# What would perfect tracking cost?
print("\n💡 THEORETICAL MINIMUM COSTS:")
print("-" * 40)
# If we perfectly tracked target, lataccel_cost = 0
# But what's the minimum achievable jerk?
all_target_jerks = []
for df in all_data:
    target = df['targetLateralAcceleration'].values
    target_jerk = np.diff(target) / DEL_T
    all_target_jerks.extend(target_jerk[CONTROL_START_IDX-1:COST_END_IDX-1])

all_target_jerks = np.array(all_target_jerks)
min_jerk_cost = np.mean(all_target_jerks**2) * 100
print(f"If we PERFECTLY tracked the target:")
print(f"  lataccel_cost = 0")
print(f"  jerk_cost = mean(target_jerk²) × 100 = {min_jerk_cost:.2f}")
print(f"  total_cost = {min_jerk_cost:.2f}")
print()
print(f"⚠️ But this assumes instant response - physically impossible!")
print(f"   The neural physics model has delay/dynamics we can't bypass.")

# =============================================================================
# PART 7: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: GENERATING VISUALIZATIONS")
print("=" * 80)

fig_dir = Path(__file__).parent / "data_analysis_figures"
fig_dir.mkdir(exist_ok=True)

# Figure 1: Global distributions
print("\nGenerating Figure 1: Global distributions...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Speed distribution
axes[0, 0].hist(combined['vEgo'], bins=100, edgecolor='none', alpha=0.7)
axes[0, 0].set_xlabel('Speed (m/s)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Vehicle Speed Distribution')
axes[0, 0].axvline(combined['vEgo'].mean(), color='r', linestyle='--', label=f'Mean: {combined["vEgo"].mean():.1f}')
axes[0, 0].legend()

# Acceleration distribution
axes[0, 1].hist(combined['aEgo'], bins=100, edgecolor='none', alpha=0.7)
axes[0, 1].set_xlabel('Acceleration (m/s²)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Vehicle Acceleration Distribution')

# Roll distribution
axes[0, 2].hist(np.degrees(combined['roll']), bins=100, edgecolor='none', alpha=0.7)
axes[0, 2].set_xlabel('Roll (degrees)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Road Roll Distribution')

# Target lataccel distribution
axes[1, 0].hist(combined['targetLateralAcceleration'], bins=100, edgecolor='none', alpha=0.7)
axes[1, 0].set_xlabel('Target Lat Accel (m/s²)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Target Lateral Acceleration Distribution')

# Steer command distribution
axes[1, 1].hist(combined['steer_command_flipped'], bins=100, edgecolor='none', alpha=0.7)
axes[1, 1].set_xlabel('Steer Command')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Steering Command Distribution')

# Roll lataccel distribution
axes[1, 2].hist(combined['roll_lataccel'], bins=100, edgecolor='none', alpha=0.7)
axes[1, 2].set_xlabel('Roll Lat Accel (m/s²)')
axes[1, 2].set_ylabel('Count')
axes[1, 2].set_title('Roll-Induced Lateral Acceleration')

plt.tight_layout()
plt.savefig(fig_dir / '01_global_distributions.png', dpi=150)
plt.close()

# Figure 2: Correlations
print("Generating Figure 2: Feature correlations...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Speed vs target lataccel
sample = combined.sample(n=50000, random_state=42)
axes[0].hexbin(sample['vEgo'], sample['targetLateralAcceleration'], 
               gridsize=50, cmap='viridis', mincnt=1)
axes[0].set_xlabel('Speed (m/s)')
axes[0].set_ylabel('Target Lat Accel (m/s²)')
axes[0].set_title('Speed vs Target Lataccel')
plt.colorbar(axes[0].collections[0], ax=axes[0], label='Count')

# Speed vs steer command
axes[1].hexbin(sample['vEgo'], sample['steer_command_flipped'], 
               gridsize=50, cmap='viridis', mincnt=1)
axes[1].set_xlabel('Speed (m/s)')
axes[1].set_ylabel('Steer Command')
axes[1].set_title('Speed vs Steer Command')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Count')

# Target lataccel vs steer command
axes[2].hexbin(sample['targetLateralAcceleration'], sample['steer_command_flipped'], 
               gridsize=50, cmap='viridis', mincnt=1)
axes[2].set_xlabel('Target Lat Accel (m/s²)')
axes[2].set_ylabel('Steer Command')
axes[2].set_title('Target Lataccel vs Steer Command')
plt.colorbar(axes[2].collections[0], ax=axes[2], label='Count')

plt.tight_layout()
plt.savefig(fig_dir / '02_correlations.png', dpi=150)
plt.close()

# Figure 3: Sample trajectories
print("Generating Figure 3: Sample trajectories...")
fig, axes = plt.subplots(4, 4, figsize=(16, 12))

np.random.seed(42)
sample_idxs = np.random.choice(len(all_data), 16, replace=False)

for i, idx in enumerate(sample_idxs):
    ax = axes[i // 4, i % 4]
    df = all_data[idx]
    
    ax.plot(df['targetLateralAcceleration'], label='Target', alpha=0.8)
    ax.axvline(CONTROL_START_IDX, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(COST_END_IDX, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'File {idx:05d}', fontsize=9)
    ax.set_ylim(-5, 5)
    if i % 4 == 0:
        ax.set_ylabel('Lat Accel (m/s²)')
    if i >= 12:
        ax.set_xlabel('Step')

plt.suptitle('16 Random Trajectories (Target Lateral Acceleration)', fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / '03_sample_trajectories.png', dpi=150)
plt.close()

# Figure 4: Trajectory statistics distribution
print("Generating Figure 4: Per-trajectory statistics...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(stats_df['target_lataccel_std'], bins=50, edgecolor='none', alpha=0.7)
axes[0, 0].set_xlabel('Target Lataccel Std')
axes[0, 0].set_title('Trajectory Difficulty (Target Variance)')

axes[0, 1].hist(stats_df['target_lataccel_range'], bins=50, edgecolor='none', alpha=0.7)
axes[0, 1].set_xlabel('Target Lataccel Range')
axes[0, 1].set_title('Target Dynamic Range per Trajectory')

axes[0, 2].hist(stats_df['target_jerk_mean'], bins=50, edgecolor='none', alpha=0.7)
axes[0, 2].set_xlabel('Mean Target Jerk')
axes[0, 2].set_title('Target Rate of Change')

axes[1, 0].hist(stats_df['v_ego_mean'], bins=50, edgecolor='none', alpha=0.7)
axes[1, 0].set_xlabel('Mean Speed (m/s)')
axes[1, 0].set_title('Average Speed per Trajectory')

axes[1, 1].hist(stats_df['v_ego_std'], bins=50, edgecolor='none', alpha=0.7)
axes[1, 1].set_xlabel('Speed Std (m/s)')
axes[1, 1].set_title('Speed Variation per Trajectory')

axes[1, 2].hist(stats_df['roll_lataccel_std'], bins=50, edgecolor='none', alpha=0.7)
axes[1, 2].set_xlabel('Roll Lataccel Std')
axes[1, 2].set_title('Road Hilliness per Trajectory')

plt.tight_layout()
plt.savefig(fig_dir / '04_trajectory_statistics.png', dpi=150)
plt.close()

# Figure 5: Time-aligned analysis
print("Generating Figure 5: Time-aligned statistics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compute mean/std at each timestep across all trajectories
# First find the minimum length trajectory
min_len = min(len(df) for df in all_data)
print(f"  Min trajectory length: {min_len} timesteps")

mean_target = np.zeros(min_len)
std_target = np.zeros(min_len)
mean_steer = np.zeros(min_len)
std_steer = np.zeros(min_len)

for i in range(min_len):
    targets = [df['targetLateralAcceleration'].iloc[i] for df in all_data if len(df) > i]
    steers = [df['steer_command_flipped'].iloc[i] for df in all_data if len(df) > i]
    mean_target[i] = np.mean(targets)
    std_target[i] = np.std(targets)
    mean_steer[i] = np.mean(steers)
    std_steer[i] = np.std(steers)

timesteps = np.arange(min_len)

axes[0, 0].fill_between(timesteps, mean_target - std_target, mean_target + std_target, alpha=0.3)
axes[0, 0].plot(timesteps, mean_target, 'b-', linewidth=2)
axes[0, 0].axvline(CONTROL_START_IDX, color='r', linestyle='--', label='Control Start')
axes[0, 0].axvline(COST_END_IDX, color='r', linestyle='--', label='Cost End')
axes[0, 0].set_xlabel('Timestep')
axes[0, 0].set_ylabel('Target Lat Accel (m/s²)')
axes[0, 0].set_title('Mean Target Lataccel ± Std (across all trajectories)')
axes[0, 0].legend()

axes[0, 1].fill_between(timesteps, mean_steer - std_steer, mean_steer + std_steer, alpha=0.3)
axes[0, 1].plot(timesteps, mean_steer, 'g-', linewidth=2)
axes[0, 1].axvline(CONTROL_START_IDX, color='r', linestyle='--')
axes[0, 1].axvline(COST_END_IDX, color='r', linestyle='--')
axes[0, 1].set_xlabel('Timestep')
axes[0, 1].set_ylabel('Steer Command')
axes[0, 1].set_title('Mean Steer Command ± Std (across all trajectories)')

# Variance over time
axes[1, 0].plot(timesteps, std_target, 'b-', linewidth=2)
axes[1, 0].axvline(CONTROL_START_IDX, color='r', linestyle='--')
axes[1, 0].axvline(COST_END_IDX, color='r', linestyle='--')
axes[1, 0].set_xlabel('Timestep')
axes[1, 0].set_ylabel('Std')
axes[1, 0].set_title('Target Lataccel Variability Over Time')

axes[1, 1].plot(timesteps, std_steer, 'g-', linewidth=2)
axes[1, 1].axvline(CONTROL_START_IDX, color='r', linestyle='--')
axes[1, 1].axvline(COST_END_IDX, color='r', linestyle='--')
axes[1, 1].set_xlabel('Timestep')
axes[1, 1].set_ylabel('Std')
axes[1, 1].set_title('Steer Command Variability Over Time')

plt.tight_layout()
plt.savefig(fig_dir / '05_time_aligned_stats.png', dpi=150)
plt.close()

# Figure 6: Frequency analysis
print("Generating Figure 6: Frequency analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target frequency content - filter to reasonable range
freq_mask = freqs < 5.0
plot_freqs = freqs[freq_mask]
plot_psd = psd[freq_mask]

axes[0].semilogy(plot_freqs, plot_psd)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Power Spectral Density')
axes[0].set_title('Target Lataccel Frequency Content')
axes[0].axvline(1.0, color='r', linestyle='--', alpha=0.5, label='1 Hz')
axes[0].axvline(2.0, color='orange', linestyle='--', alpha=0.5, label='2 Hz')
axes[0].legend()
axes[0].set_xlim(0, 5)

# Cumulative power
cumsum = np.cumsum(plot_psd)
cumsum = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
axes[1].plot(plot_freqs, cumsum)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Cumulative Power (normalized)')
axes[1].set_title('Cumulative Power Distribution')
axes[1].axhline(0.9, color='r', linestyle='--', label='90% power')
axes[1].axhline(0.99, color='orange', linestyle='--', label='99% power')
axes[1].legend()
axes[1].set_xlim(0, 5)

plt.tight_layout()
plt.savefig(fig_dir / '06_frequency_analysis.png', dpi=150)
plt.close()

# Figure 7: Extreme cases
print("Generating Figure 7: Extreme cases comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Easiest trajectory
easiest_idx = stats_df['target_lataccel_std'].idxmin()
easiest_df = all_data[easiest_idx]
axes[0, 0].plot(easiest_df['targetLateralAcceleration'], label='Target', color='blue')
axes[0, 0].plot(easiest_df['steer_command_flipped'], label='Steer Cmd', color='green', alpha=0.7)
axes[0, 0].axvline(CONTROL_START_IDX, color='gray', linestyle='--')
axes[0, 0].axvline(COST_END_IDX, color='gray', linestyle='--')
axes[0, 0].set_title(f'EASIEST Trajectory (File {easiest_idx:05d})')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()
axes[0, 0].set_ylim(-5, 5)

# Hardest trajectory
hardest_idx = stats_df['target_lataccel_std'].idxmax()
hardest_df = all_data[hardest_idx]
axes[0, 1].plot(hardest_df['targetLateralAcceleration'], label='Target', color='blue')
axes[0, 1].plot(hardest_df['steer_command_flipped'], label='Steer Cmd', color='green', alpha=0.7)
axes[0, 1].axvline(CONTROL_START_IDX, color='gray', linestyle='--')
axes[0, 1].axvline(COST_END_IDX, color='gray', linestyle='--')
axes[0, 1].set_title(f'HARDEST Trajectory (File {hardest_idx:05d})')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].set_ylim(-5, 5)

# Highest speed trajectory
highest_speed_idx = stats_df['v_ego_mean'].idxmax()
highest_speed_df = all_data[highest_speed_idx]
axes[1, 0].plot(highest_speed_df['vEgo'], label='Speed', color='orange')
ax2 = axes[1, 0].twinx()
ax2.plot(highest_speed_df['targetLateralAcceleration'], label='Target', color='blue', alpha=0.7)
axes[1, 0].axvline(CONTROL_START_IDX, color='gray', linestyle='--')
axes[1, 0].axvline(COST_END_IDX, color='gray', linestyle='--')
axes[1, 0].set_title(f'HIGHEST SPEED (File {highest_speed_idx:05d})')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Speed (m/s)', color='orange')
ax2.set_ylabel('Target Lataccel (m/s²)', color='blue')

# Hilliest trajectory
hilliest_idx = stats_df['roll_lataccel_std'].idxmax()
hilliest_df = all_data[hilliest_idx]
axes[1, 1].plot(hilliest_df['roll_lataccel'], label='Roll Lataccel', color='purple')
axes[1, 1].plot(hilliest_df['targetLateralAcceleration'], label='Target', color='blue', alpha=0.7)
axes[1, 1].axvline(CONTROL_START_IDX, color='gray', linestyle='--')
axes[1, 1].axvline(COST_END_IDX, color='gray', linestyle='--')
axes[1, 1].set_title(f'HILLIEST Trajectory (File {hilliest_idx:05d})')
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('Lat Accel (m/s²)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(fig_dir / '07_extreme_cases.png', dpi=150)
plt.close()

print(f"\n✅ All figures saved to: {fig_dir}")

# =============================================================================
# PART 8: KEY INSIGHTS SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PART 8: KEY INSIGHTS SUMMARY")
print("=" * 80)

print("""
🔍 DATA TEXTURE - WHAT WE LEARNED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SCALE
   • 20,000 trajectories × 600 timesteps = 12 million data points
   • Each trajectory is 60 seconds @ 10 Hz
   • Control window: steps 100-500 (40 seconds of evaluation)

2. INPUTS (what the controller sees)
   • Target lateral acceleration: [-4.9, 4.6] m/s² (roughly ±0.5g)
   • Vehicle speed: [0, 40] m/s (0-144 km/h)
   • Vehicle acceleration: [-4, 4] m/s²
   • Roll-induced lat accel: up to ±0.5 m/s² (disturbance!)
   • Future plan: 50 steps (5 seconds) of targets ahead

3. OUTPUTS (what the controller produces)
   • Steer command: [-2, 2] (clipped)
   • Must track target while minimizing jerk

4. WHY THIS IS HARD:
   
   a) NONLINEAR DYNAMICS
      • Physics model is a neural network (black box)
      • Steer → lataccel relationship is complex, nonlinear
      • Speed affects the gain significantly
   
   b) DELAYED RESPONSE
      • 20-step context history matters
      • Actions don't have instant effect
      • Must predict/anticipate
   
   c) DISTURBANCES
      • Road roll adds up to 0.5 m/s² of unmeasured disturbance
      • Vehicle dynamics (aEgo) affect response
   
   d) JERK PENALTY
      • Can't just bang-bang between targets
      • Must smooth the control
      • Trade-off between tracking and smoothness

5. TRAJECTORY DIVERSITY:
   • Easy: nearly constant targets (highway straight)
   • Hard: high-frequency oscillations (winding roads)
   • Speed range: 0-40 m/s with high variance
   • Some trajectories are 10× harder than others

6. COST FUNCTION:
   • Total = 50 × lataccel_cost + jerk_cost
   • Tracking is 50× more important than smoothness
   • But neural physics has delay, so perfect tracking impossible
   • Theoretical minimum (perfect tracking): jerk_cost ≈ {min_jerk_cost:.1f}

7. BASELINE PERFORMANCE:
   • PID controller: ~107 total cost
   • Best BC (exp023): ~103 total cost
   • Target: <45 total cost (56% improvement needed!)

8. KEY CORRELATIONS:
   • Speed ↔ Steer sensitivity (high speed = less steering needed)
   • Target ↔ Steer: negative correlation (counterintuitive!)
   • Roll is a disturbance, not a control input
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nFigures saved to: {fig_dir}")
print(f"Total trajectories analyzed: {len(all_data)}")
print(f"Total data points: {len(all_data) * 600:,}")
