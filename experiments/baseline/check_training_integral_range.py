#!/usr/bin/env python3
"""
Check: What error_integral range did BC see during training?

Method: Run PID on training files and log error_integral distribution.
"""

import numpy as np
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid

print("\n" + "="*60)
print("TRAINING DATA INTEGRAL RANGE ANALYSIS")
print("="*60)

# Use same files as BC training (first 5000 sorted)
all_files = sorted(glob.glob("./data/*.csv"))
np.random.seed(42)
np.random.shuffle(all_files)
train_files = all_files[:int(0.9 * len(all_files))]
train_sample = train_files[:1000]  # Sample 1000 for speed

model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)

print(f"\nSampling {len(train_sample)} training files...")

class InstrumentedPID:
    def __init__(self):
        self.pid = pid.Controller()
        self.integrals = []
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.integrals.append(self.pid.error_integral)
        return self.pid.update(target_lataccel, current_lataccel, state, future_plan)

all_integrals = []

for data_file in tqdm(train_sample, desc="Collecting integrals"):
    controller = InstrumentedPID()
    sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
    sim.rollout()
    all_integrals.extend(controller.integrals)

all_integrals = np.array(all_integrals)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nTotal data points: {len(all_integrals):,}")
print(f"\nError integral distribution (PID on training data):")
print(f"  Mean: {np.mean(all_integrals):.2f}")
print(f"  Std:  {np.std(all_integrals):.2f}")
print(f"  Min:  {np.min(all_integrals):.2f}")
print(f"  Max:  {np.max(all_integrals):.2f}")

percentiles = [0.1, 1, 5, 95, 99, 99.9]
pct_values = np.percentile(all_integrals, percentiles)

print(f"\nPercentiles:")
for p, v in zip(percentiles, pct_values):
    print(f"  {p:5.1f}%: {v:7.2f}")

print(f"\nOut-of-distribution threshold suggestions:")
threshold_999 = np.max(np.abs([pct_values[0], pct_values[-1]]))
print(f"  99.9% coverage: ±{threshold_999:.1f}")

threshold_99 = np.max(np.abs([pct_values[1], pct_values[-2]]))
print(f"  99% coverage:   ±{threshold_99:.1f}")

threshold_95 = np.max(np.abs([pct_values[2], pct_values[-3]]))
print(f"  95% coverage:   ±{threshold_95:.1f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print(f"\nFile 00069 integral values:")
print(f"  BC reached:  ±47.8")
print(f"  PID reached: ±7.1")

if 47.8 > threshold_999:
    print(f"\n❌ BC's integral (±47.8) is WAY beyond training range (±{threshold_999:.1f})")
    print("   This explains the catastrophic failure!")
    print(f"\n✅ SOLUTION: Clamp error_integral to ±{threshold_999:.0f} (99.9% coverage)")
    print("   This keeps BC in-distribution while allowing normal operation")
else:
    print(f"\n⚠️  BC's integral (±47.8) is within training range")
    print("   Need to investigate other causes")

print("="*60)

