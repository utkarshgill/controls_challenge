"""Debug: compare RNG values between official sim and batched sim."""

import numpy as np, sys, os
from hashlib import md5
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Official sim path
data_path = "data/00000.csv"
seed_official = int(md5(data_path.encode()).hexdigest(), 16) % 10**4
print(f"Official seed for '{data_path}': {seed_official}")

# Batched sim path
seed_prefix = os.getenv("SEED_PREFIX", "data")
fname = "00000.csv"
seed_str = f"{seed_prefix}/{fname}"
seed_batched = int(md5(seed_str.encode()).hexdigest(), 16) % 10**4
print(f"Batched seed for '{seed_str}': {seed_batched}")
print(f"Seeds match: {seed_official == seed_batched}")

# Now check: does np.random.choice consume the same RNG as np.random.rand?
np.random.seed(seed_official)
# The official sim does reset() which calls np.random.seed(seed)
# Then for each step from CONTEXT_LENGTH to T, it calls model.predict()
# which calls np.random.choice(1024, p=probs)

# Let's see what np.random.choice actually does under the hood
np.random.seed(42)
vals_choice = []
for _ in range(10):
    # Simulate what np.random.choice does for uniform-ish probs
    probs = np.ones(1024) / 1024
    c = np.random.choice(1024, p=probs)
    vals_choice.append(c)

np.random.seed(42)
vals_rand = []
for _ in range(10):
    u = np.random.rand()
    vals_rand.append(u)

print(f"\nnp.random.choice tokens: {vals_choice[:5]}")
print(f"np.random.rand values:   {vals_rand[:5]}")

# Check: does choice(p=probs) consume exactly 1 rand call?
np.random.seed(42)
_ = np.random.choice(1024, p=np.ones(1024) / 1024)
state_after_choice = np.random.get_state()[1][:5].copy()

np.random.seed(42)
_ = np.random.rand()
state_after_rand = np.random.get_state()[1][:5].copy()

print(f"\nRNG state after choice: {state_after_choice}")
print(f"RNG state after rand:   {state_after_rand}")
print(f"States match: {np.array_equal(state_after_choice, state_after_rand)}")

# Also check: batched sim pre-generates rng.rand(n_steps)
# Does this match calling rng.rand() one at a time?
rng1 = np.random.RandomState(seed_official)
bulk = rng1.rand(10)

rng2 = np.random.RandomState(seed_official)
sequential = [rng2.rand() for _ in range(10)]

print(f"\nBulk rand:       {bulk[:5]}")
print(f"Sequential rand: {sequential[:5]}")
print(f"Bulk == Sequential: {np.allclose(bulk, sequential)}")

# Now the critical question: does np.random.choice(p=probs) use the
# SAME rng consumption as a single rand() call?
# Let's test with actual CDF-based sampling
np.random.seed(seed_official)
probs = np.ones(1024) / 1024
choice_result = np.random.choice(1024, p=probs)
state_after_1 = np.random.get_state()

np.random.seed(seed_official)
u = np.random.rand()
# Manual choice: cdf lookup
cdf = np.cumsum(probs)
manual_choice = np.searchsorted(cdf, u)
state_after_2 = np.random.get_state()

print(f"\nnp.random.choice result: {choice_result}")
print(f"Manual cdf choice:       {manual_choice}")
print(f"rand value used:         {u:.10f}")
print(
    f"RNG states match after 1 call: {np.array_equal(state_after_1[1], state_after_2[1])}"
)
