# Experiment XXX: [Name]

**Date**: YYYY-MM-DD  
**Status**: 🏃 Running | ✅ Complete | ❌ Failed  
**Researcher**: [Your name]

---

## Hypothesis
What are we testing? What do we expect to happen?

---

## Motivation
Why is this worth trying? What evidence supports this direction?

---

## Method

### Model Architecture
- Network: [e.g., 3-layer MLP, 128 hidden]
- Input dim: [e.g., 57]
- Output: [e.g., continuous action ∈ [-2, 2]]

### State Representation
```
Features: [list all features]
OBS_SCALE: [normalization factors]
```

### Training
- Dataset: [e.g., 5000 files]
- Epochs: [e.g., 50]
- Batch size: [e.g., 256]
- Learning rate: [e.g., 1e-3]
- Other hyperparams: [...]

### Evaluation
- Test set: [e.g., 100 files]
- Metrics: [mean, median, worst-case]

---

## Results

### Quantitative
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Mean cost | XX.X | ±X.X |
| Median cost | XX.X | ±X.X |
| Std dev | XX.X | ±X.X |
| Best file | XX.X | ±X.X |
| Worst file | XX.X | ±X.X |
| Failures (>2× median) | X/100 | ±X |

### Comparison to Baselines
```
PID:  80.4
BC:   92.4
This: XX.X  [← Better/Worse by X.X%]
```

### Key Observations
- [Observation 1]
- [Observation 2]
- [Observation 3]

---

## Analysis

### What Worked
- [What went well]

### What Didn't Work
- [What failed]

### Surprising Findings
- [Unexpected results]

---

## Conclusion

**Did the hypothesis hold?** Yes/No/Partially

**Why?** [Explanation]

**Key takeaway**: [One sentence summary]

---

## Next Steps

Based on these results:
1. [Next experiment idea 1]
2. [Next experiment idea 2]
3. [Alternative approach if this failed]

---

## Reproducibility

### Command to reproduce:
```bash
cd experiments/expXXX_name
python run.py --config config.yaml
```

### Dependencies:
- Python 3.x
- PyTorch x.x
- [Other deps]

### Compute:
- Time: [e.g., 2 hours on M1 Mac]
- Resources: [e.g., 16 CPU cores]

---

## Artifacts

- Checkpoint: `results/checkpoints/best.pt`
- Metrics: `results/metrics.json`
- Plots: `results/plots/`
- Logs: `results/logs/train.log`
