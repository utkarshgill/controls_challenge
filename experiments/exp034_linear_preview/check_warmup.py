#!/usr/bin/env python3
"""Check if warmup is cutting off episode"""
import pandas as pd

data_path = "../../data/00000.csv"
df = pd.read_csv(data_path)

print(f"Total data length: {len(df)} steps")
print(f"Warmup: 50 steps")
print(f"After warmup: {len(df) - 50} steps remaining")
print(f"\nControl starts at step 100 (hardcoded in simulator)")
print(f"Cost measured from 100 to 500")
print(f"So warmup of 50 means we start residual at step 50")
print(f"But control doesn't start until step 100 anyway")
print(f"\nThis seems wrong - warmup should start at step 100, not step 0")

