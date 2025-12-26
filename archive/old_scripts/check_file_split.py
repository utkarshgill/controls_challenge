#!/usr/bin/env python3
"""Check if file 00069 was in training set"""

import glob
import numpy as np

all_files = sorted(glob.glob('./data/*.csv'))
np.random.seed(42)
np.random.shuffle(all_files)
train_files = set(all_files[:int(0.9 * len(all_files))])

file_00069 = './data/00069.csv'
file_00000 = './data/00000.csv'

print(f"File 00000 (easy): {'TRAIN' if file_00000 in train_files else 'VAL'}")
print(f"File 00069 (hard): {'TRAIN' if file_00069 in train_files else 'VAL'}")

print(f"\nFirst 100 sorted files (our test set):")
first_100 = sorted(glob.glob('./data/*.csv'))[:100]
in_train = sum(1 for f in first_100 if f in train_files)
print(f"  In training set: {in_train}/100 ({100*in_train/100:.0f}%)")
print(f"  In validation set: {100-in_train}/100 ({100*(100-in_train)/100:.0f}%)")

