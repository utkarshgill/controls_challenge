#!/usr/bin/env python3
"""
Automated Migration Script

Organizes the project structure according to MIGRATION_GUIDE.md
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

def safe_move(src, dst, dry_run=True):
    """Safely move file/directory"""
    src_path = ROOT / src
    dst_path = ROOT / dst
    
    if not src_path.exists():
        print(f"⏭️  Skip: {src} (doesn't exist)")
        return
    
    # Create destination directory
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        print(f"📋 Would move: {src} → {dst}")
    else:
        try:
            shutil.move(str(src_path), str(dst_path))
            print(f"✅ Moved: {src} → {dst}")
        except Exception as e:
            print(f"❌ Error moving {src}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate project structure")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be done without doing it")
    parser.add_argument("--execute", action="store_true",
                       help="Actually perform the migration")
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN - No files will be moved")
        print("Run with --execute to actually perform migration")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("EXECUTING MIGRATION")
        print("="*60 + "\n")
    
    # Phase 2: Organize Scripts
    print("\n[Phase 2] Moving analysis scripts to scripts/")
    scripts_to_move = [
        "analyze_failures.py",
        "test_a_ego_hypothesis.py",
        "check_file_split.py",
        "check_pid.py",
        "check_training_distribution.py",
        "check_training_integral_range.py",
        "diagnose_bc_vs_pid.py",
        "evaluate_bc_quality.py",
        "verify_fix.py",
        "test_antiwindup_fix.py",
        "test_antiwindup_full.py",
        "test_curvature_fix.py",
        "test_what_we_have.py",
        "eval_ppo_simple.py",
        "eval.py",
        "baseline.py",
        "final_baseline.py",
        "final_diagnosis.py",
        "final_evaluation.py",
        "back_to_basics.py",
    ]
    
    for script in scripts_to_move:
        safe_move(script, f"scripts/{script}", dry_run)
    
    # Phase 3: Organize Documentation
    print("\n[Phase 3] Moving documentation to docs/")
    docs_to_move = [
        "FINDINGS_SUMMARY.md",
        "EXPERIMENT_PLAN.md",
        "PROJECT_STRUCTURE.md",
        "MIGRATION_GUIDE.md",
    ]
    
    for doc in docs_to_move:
        safe_move(doc, f"docs/{doc}", dry_run)
    
    docs_to_archive = [
        "BC_SUMMARY.md",
        "PROGRESS.md",
        "STATUS.md",
        "STRUCTURE.md",
        "WHAT_WORKS.md",
        "PARALLEL_REFACTOR.md",
        "PIPELINE_GUIDE.md",
    ]
    
    for doc in docs_to_archive:
        safe_move(doc, f"archive/old_docs/{doc}", dry_run)
    
    # Phase 4: Clean Up Experiments
    print("\n[Phase 4] Archiving old experiment folders")
    old_experiments = [
        "exp01_bc_baseline",
        "exp02_bc_with_a_ego",
    ]
    
    for exp in old_experiments:
        safe_move(f"experiments/{exp}", f"archive/old_experiments/{exp}", dry_run)
    
    # Phase 5: Archive Old Attempts (already in archive, just rename)
    print("\n[Phase 5] Organizing archive")
    # attempts/ and attempts_2/ are already there
    
    # Phase 7: Clean Up Checkpoints
    print("\n[Phase 7] Moving checkpoints to experiment folders")
    checkpoints = [
        ("bc_pid_best.pth", "experiments/baseline/results/checkpoints/bc_pid_best.pth"),
        ("bc_pid_checkpoint.pth", "experiments/baseline/results/checkpoints/bc_pid_checkpoint.pth"),
        ("ppo_parallel_best.pth", "experiments/baseline/results/checkpoints/ppo_parallel_best.pth"),
    ]
    
    for src, dst in checkpoints:
        safe_move(src, dst, dry_run)
    
    # Phase 7: Move results
    print("\n[Phase 7] Moving results to baseline experiment")
    results = [
        ("final_results.npz", "experiments/baseline/results/final_results.npz"),
        ("baseline_results.npz", "experiments/baseline/results/baseline_results.npz"),
    ]
    
    for src, dst in results:
        safe_move(src, dst, dry_run)
    
    # Phase 8: Clean Up Config Files
    print("\n[Phase 8] Archiving old config files")
    safe_move("experiment_config.py", "archive/experiment_config.py", dry_run)
    
    # Summary
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN COMPLETE")
        print("\nTo execute the migration, run:")
        print("  python scripts/migrate_structure.py --execute")
    else:
        print("MIGRATION COMPLETE!")
        print("\nNext steps:")
        print("  1. Verify everything works")
        print("  2. Run: python scripts/manage_experiments.py list")
        print("  3. Start Experiment 001!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

