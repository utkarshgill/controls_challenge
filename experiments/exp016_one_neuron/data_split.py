"""
Data Split Configuration for Experiment 016

Total: 20,000 files
- Train: 0-14,999 (15,000 files) - 75%
- Val:   15,000-17,499 (2,500 files) - 12.5%
- Test:  17,500-19,999 (2,500 files) - 12.5%

This ensures NO data leakage between train/val/test.
"""
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def get_data_split():
    """Get train/val/test file lists"""
    all_files = sorted([str(f) for f in DATA_DIR.glob('*.csv')])
    
    if len(all_files) != 20000:
        print(f"WARNING: Expected 20,000 files, found {len(all_files)}")
    
    train_files = all_files[0:15000]
    val_files = all_files[15000:17500]
    test_files = all_files[17500:20000]
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files,
        'all': all_files
    }

def print_split_info():
    """Print split statistics"""
    split = get_data_split()
    
    print("="*80)
    print("Data Split Configuration")
    print("="*80)
    print(f"Train: {len(split['train']):,} files (indices 0-14,999)")
    print(f"Val:   {len(split['val']):,} files (indices 15,000-17,499)")
    print(f"Test:  {len(split['test']):,} files (indices 17,500-19,999)")
    print(f"Total: {len(split['all']):,} files")
    print("="*80)
    
    # Show first/last file of each split
    print(f"\nTrain: {Path(split['train'][0]).name} ... {Path(split['train'][-1]).name}")
    print(f"Val:   {Path(split['val'][0]).name} ... {Path(split['val'][-1]).name}")
    print(f"Test:  {Path(split['test'][0]).name} ... {Path(split['test'][-1]).name}")
    print()

if __name__ == '__main__':
    print_split_info()



