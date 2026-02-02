"""Analyze route characteristics"""
import pandas as pd
import numpy as np
from pathlib import Path
import random

all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_files = all_files[17500:17520]  # First 20 test routes

results = []
for f in test_files:
    df = pd.read_csv(f)
    
    v_mean = df['vEgo'].mean()
    v_std = df['vEgo'].std()
    v_min = df['vEgo'].min()
    v_max = df['vEgo'].max()
    
    lat_mean = df['targetLateralAcceleration'].abs().mean()
    lat_std = df['targetLateralAcceleration'].std()
    lat_max = df['targetLateralAcceleration'].abs().max()
    
    # Compute curvature
    curvs = (df['targetLateralAcceleration'] - df['roll']) / np.maximum(df['vEgo'] ** 2, 1.0)
    curv_mean = curvs.abs().mean()
    curv_max = curvs.abs().max()
    
    # Check for turns/maneuvers
    num_high_lat = (df['targetLateralAcceleration'].abs() > 1.0).sum()
    
    results.append({
        'file': f.name,
        'v_mean': v_mean,
        'v_range': v_max - v_min,
        'lat_mean': lat_mean,
        'lat_max': lat_max,
        'curv_mean': curv_mean,
        'curv_max': curv_max,
        'num_high_lat': num_high_lat
    })

# Sort by different criteria
print("Route characteristics:\n")
for i, r in enumerate(results):
    print(f"{i+1}. {r['file'][:15]:15s}: v={r['v_mean']:4.1f}±{r['v_range']:4.1f} "
          f"lat={r['lat_mean']:.3f}(max={r['lat_max']:.2f}) "
          f"curv={r['curv_mean']:.4f}(max={r['curv_max']:.3f}) "
          f"high_lat={r['num_high_lat']}")

print(f"\nAverages:")
print(f"  v_mean: {np.mean([r['v_mean'] for r in results]):.1f}")
print(f"  lat_mean: {np.mean([r['lat_mean'] for r in results]):.3f}")
print(f"  curv_mean: {np.mean([r['curv_mean'] for r in results]):.4f}")

