#!/usr/bin/env bash
set -euo pipefail

ORT_NIGHTLY_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-13.1/lib64}"

# 1. Generic deps first.
uv pip install -r requirements.txt

# 2. Nuke any existing ORT build so the GPU build is authoritative.
uv pip uninstall onnxruntime onnxruntime-gpu || true

# 3. GPU-specific: TensorRT libs + nightly onnxruntime-gpu (CUDA 13 + TRT provider).
uv pip install tensorrt-cu13
uv pip install --pre --extra-index-url "${ORT_NIGHTLY_INDEX}" onnxruntime-gpu

TRT_LIB_DIR="$(
python - <<'PY'
from pathlib import Path
import site

for base in site.getsitepackages():
    p = Path(base) / "tensorrt_libs"
    if p.exists():
        print(p)
        break
else:
    raise SystemExit("tensorrt_libs not found in site-packages")
PY
)"

# 4. LD_LIBRARY_PATH for TRT + CUDA (idempotent append).
grep -q 'tensorrt_libs' ~/.bashrc 2>/dev/null || \
  echo "export LD_LIBRARY_PATH=\"${CUDA_LIB_DIR}:${TRT_LIB_DIR}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc

export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${TRT_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# 5. Verify.
python - <<'PY'
import onnxruntime as ort

provs = ort.get_available_providers()
print("Providers:", provs)
assert 'CUDAExecutionProvider' in provs, 'CUDA provider missing!'
assert 'TensorrtExecutionProvider' in provs, 'TensorRT provider missing!'
PY

echo "Done. CUDA + TensorRT providers available."
