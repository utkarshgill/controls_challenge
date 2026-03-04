#!/usr/bin/env bash
set -e

# 1. Generic deps first (onnxruntime CPU listed here gets installed then overwritten)
uv pip install -r requirements.txt

# 2. Nuke any CPU onnxruntime so it can't shadow the GPU build
uv pip uninstall onnxruntime onnxruntime-gpu || true

# 3. GPU-specific: TensorRT libs + nightly onnxruntime-gpu (CUDA 13 + TRT provider)
uv pip install tensorrt-cu13
uv pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu

# 4. LD_LIBRARY_PATH for TRT + CUDA (idempotent append)
grep -q 'tensorrt_libs' ~/.bashrc 2>/dev/null || \
  echo 'export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:/venv/main/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}"' >> ~/.bashrc

# 5. Verify
source ~/.bashrc
python -c "import onnxruntime as ort; provs = ort.get_available_providers(); print('Providers:', provs); assert 'CUDAExecutionProvider' in provs, 'CUDA provider missing!'"

echo "Done. All providers available."