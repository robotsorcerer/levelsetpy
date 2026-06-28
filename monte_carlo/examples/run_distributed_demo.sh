#!/bin/bash
# Distributed murmuration demo launcher
#
# Usage:
#   ./examples/run_distributed_demo.sh         # Auto-detect GPU count
#   ./examples/run_distributed_demo.sh 2       # Use 2 GPUs
#   ./examples/run_distributed_demo.sh 4       # Use 4 GPUs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( dirname "$SCRIPT_DIR" )"

# Detect GPU count or use argument
if [ $# -eq 0 ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
else
    NUM_GPUS=$1
fi

echo "========================================================================"
echo "PyTorch Murmuration Distributed Demo"
echo "========================================================================"
echo "Running on $NUM_GPUS GPU(s)"
echo ""

# Adjust problem size based on GPU count
BIRDS=$((100000 * NUM_GPUS))
SAMPLES=$((5000))

echo "Configuration:"
echo "  Total birds: $BIRDS"
echo "  MC samples: $SAMPLES"
echo "  Time steps: 3"
echo "  Flocks: 5"
echo "  Predators: 2"
echo ""
echo "Command:"
echo "  torchrun --nproc_per_node=$NUM_GPUS \\"
echo "    examples/ex_murmuration_torch.py \\"
echo "    --n-birds $BIRDS --n-samples $SAMPLES"
echo ""
echo "========================================================================"
echo ""

# Run with torchrun
cd "$REPO_DIR"
torchrun --nproc_per_node="$NUM_GPUS" \
  examples/ex_murmuration_torch.py \
  --n-birds "$BIRDS" \
  --n-flocks 5 \
  --n-predators 2 \
  --time-steps 3 \
  --max-iters 5 \
  --n-samples "$SAMPLES" \
  --grid-res 32 \
  --save-results \
  --out-dir /tmp/murmurs_torch_distributed/

echo ""
echo "========================================================================"
echo "Demo complete. Results saved to /tmp/murmurs_torch_distributed/"
echo "========================================================================"
