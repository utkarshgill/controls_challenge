"""Modify the ONNX model to only output the last timestep.
Original: output shape (N, 20, 1024)
Modified: output shape (N, 1024) — only the last position.
This reduces output memory by 20x."""

import onnx
from pathlib import Path
from onnx import helper, TensorProto
import numpy as np
import sys

model_path = "models/tinyphysics.onnx"
out_path = "models/tinyphysics_last.onnx"

model = onnx.load(model_path)
graph = model.graph

# Find the output node
print(f"Original outputs: {[o.name for o in graph.output]}")
for o in graph.output:
    print(f"  {o.name}: {o.type}")

# Add a Gather node to select the last timestep (index -1 = 19)
# output_name → Gather(axis=1, indices=19) → new_output
orig_output = graph.output[0]
orig_name = orig_output.name

# Create index tensor (scalar: 19 for last of 20 timesteps)
index_name = "last_index"
index_tensor = helper.make_tensor(index_name, TensorProto.INT64, [], [19])
graph.initializer.append(index_tensor)

# Create Gather node: output[:, 19, :] → (N, 1024)
gather_name = orig_name + "_last"
gather_node = helper.make_node(
    "Gather",
    inputs=[orig_name, index_name],
    outputs=[gather_name],
    axis=1,
)
graph.node.append(gather_node)

# Replace the output
# Remove old output, add new one
graph.output.remove(orig_output)
new_output = helper.make_tensor_value_info(gather_name, TensorProto.FLOAT, None)
graph.output.append(new_output)

# Save
onnx.save(model, out_path)
print(f"Saved modified model to {out_path}")
print(f"New output: {gather_name}")

# Verify
model2 = onnx.load(out_path)
print(f"Outputs: {[o.name for o in model2.graph.output]}")
