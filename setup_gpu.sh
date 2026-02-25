#!/usr/bin/env bash
set -e
uv pip uninstall onnxruntime onnxruntime-gpu || true

uv pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
uv pip install --pre -r requirements.txt
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:/venv/main/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}"' >> ~/.bashrc

echo "Done. Run 'source ~/.bashrc' or open a new terminal for LD_LIBRARY_PATH to take effect."