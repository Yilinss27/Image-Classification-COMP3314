#!/bin/bash
# Check what Python packages are available on this machine.
for mod in numpy pandas sklearn joblib skimage PIL cupy kaggle; do
  v=$(python -c "import ${mod}; print(${mod}.__version__)" 2>&1 | tail -1)
  echo "${mod}: ${v}"
done
echo "---"
which kaggle || echo "kaggle CLI: not in PATH"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
