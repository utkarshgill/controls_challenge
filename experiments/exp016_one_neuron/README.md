# Experiment 016: One-Neuron Controller

## Hypothesis
The simplest possible learnable controller: **1 neuron, 3 weights, no bias**

```
steering = w1 * lateral_accel_error + w2 * v_ego + w3 * current_lataccel
```

## Why This Might Work
- PID is essentially a linear controller
- Maybe a simple linear model can capture the essence
- Occam's Razor: simplest solution first

## Training
1. Collect PID demonstrations on training routes
2. Train 1-neuron network to minimize squared error
3. That's it.

## Evaluation
Use official evaluation from main README:
```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
                     --data_path ./data \
                     --controller exp016_one_neuron \
                     --num_segs 5000
```

## Expected Outcome
- Best case: ~200 cost (worse than PID but learns something)
- Realistic: Shows that simple linear model can work
- Worst case: Complete failure, proves we need more capacity

Either way, we learn something definitive.



