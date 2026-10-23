# CS 6362 Assignment 2: Paper Reproductions

PyTorch experiments for three ICLR 2017 papers. The four submission notebooks contain their actual executed outputs.

Repository: [https://github.com/yangchenxi000/cs6362assignment](https://github.com/yangchenxi000/cs6362assignment).

## Notebooks

| File | Experiment |
|---|---|
| `paper1.ipynb` | Zhang et al.: original 18-run Figure 1a/b/c protocol, including Inception, AlexNet and MLP. |
| `paper1_lr_followup.ipynb` | Zhang et al.: supplemental Figure 1b/c experiments for AlexNet and a one-hidden-layer MLP. |
| `paper2_full.ipynb` | Keskar et al.: Figure 3, F1/C1/C2, small-batch versus large-batch training and parameter interpolation. |
| `paper3_full.ipynb` | Higgins et al.: Figure 7A/B, beta-VAE with beta=4 versus VAE with beta=1 on dSprites. |

Paper 1 uses two separately reported protocols. The original notebook contains 18 runs with a 200-epoch budget and no learning-rate floor; 11 reached the strict zero-training-error event. It supplies Figure 1a and the original Figure 1b/c measurements. The follow-up contains ten AlexNet/MLP corruption runs, with a learning-rate floor of 0.001 and a maximum of 400 epochs. Nine follow-up runs reached zero training error; AlexNet with fully randomized labels ended with one incorrect training example and is marked censored. Inception was not rerun in the follow-up. Do not combine the two protocols' normalized convergence times as a single unchanged experiment.

Paper 2 uses one seed. Paper 3 uses three seeds per beta. Differences from the original papers are discussed in the reports and notebook text.

## Environment

The recorded server environment was Linux, Python 3.11.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and NumPy 2.4.6, with an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (approximately 95 GiB available). A CUDA GPU is required by the supplied execution settings.

For a new environment:

```bash
conda create -n cs6362 python=3.11 -y
conda activate cs6362
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.4.6 pandas matplotlib pillow ipython \
  nbformat nbclient nbconvert ipykernel
export CS6362_PYTHON="$(python -c 'import sys; print(sys.executable)')"
```

On the original ROSALIND environment, the existing interpreter can be used instead:

```bash
export CS6362_PYTHON=/home/NAShome/yangc21/miniconda3/envs/tripath_flow/bin/python
```

Register one kernel for all four notebooks. This overrides their recorded, machine-specific kernel names during execution:

```bash
"$CS6362_PYTHON" -m ipykernel install --user \
  --name cs6362-repro --display-name "CS6362 reproduction"
```

The extra Python packages above are not fully version-locked; the versions listed as recorded are taken from the executed notebooks. Run all following commands from the repository root.

## Download data

Prepare official MNIST and CIFAR-10 train/test splits on the login node before submitting GPU jobs. Paper 2 has automatic downloads disabled, so this preparation is required.

```bash
"$CS6362_PYTHON" - <<'PY'
from torchvision.datasets import CIFAR10, MNIST
for train in (True, False):
    CIFAR10(root="./paper1_data", train=train, download=True)
    MNIST(root="./paper1_data", train=train, download=True)
PY
```

Download the official [dSprites dataset](https://github.com/deepmind/dsprites-dataset):

```bash
mkdir -p paper3_data
curl --fail --location --retry 5 \
  'https://raw.githubusercontent.com/deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz' \
  -o paper3_data/dsprites.npz.part
mv paper3_data/dsprites.npz.part paper3_data/dsprites.npz
```

Paper 3 checks the binary images and factor-class array, then creates a packed image cache. The full dataset has 737,280 images, each 64 x 64 pixels. Ground-truth factors are used for analysis, not as training targets.

## Execute with Slurm

The following example uses the ROSALIND `gpus` partition. Adjust partition, time and resources for another cluster. It writes rerun notebooks to `executed/`, preserving the submitted notebooks and their outputs.

```bash
mkdir -p logs executed
cat > run_notebook.slurm <<'SLURM'
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${CS6362_PYTHON:?Set CS6362_PYTHON before submission}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export DATA_ROOT="$PWD/paper1_data"
export PAPER2_OUTPUT="$PWD/paper2_figure3_outputs_v2"
export DSPRITES_NPZ="$PWD/paper3_data/dsprites.npz"
export BETA_VAE_OUTPUT_ROOT="$PWD/paper3_figure7_outputs_v2"
export BETA_VAE_MODE=full
export BETA_VAE_TRAIN=1
export BETA_VAE_ANALYZE=1

case "${1:-}" in
  paper1) notebook=paper1.ipynb ;;
  paper1_followup) notebook=paper1_lr_followup.ipynb ;;
  paper2) notebook=paper2_full.ipynb ;;
  paper3) notebook=paper3_full.ipynb ;;
  *) echo "Usage: sbatch run_notebook.slurm paper1|paper1_followup|paper2|paper3"; exit 2 ;;
esac

"$CS6362_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU training"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

exec "$CS6362_PYTHON" -m jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.kernel_name=cs6362-repro \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir=executed --output="${notebook%.ipynb}_rerun.ipynb" \
  "$notebook"
SLURM

sbatch --export=ALL --job-name=paper1 run_notebook.slurm paper1
sbatch --export=ALL --job-name=paper1_followup run_notebook.slurm paper1_followup
sbatch --export=ALL --job-name=paper2 run_notebook.slurm paper2
sbatch --export=ALL --job-name=paper3 run_notebook.slurm paper3
```

These commands submit four independent GPU jobs. The 12-hour limit is an allocation limit, not a guaranteed runtime. Re-submit the same job if interrupted; compatible epoch checkpoints are reused. Existing compatible final checkpoints may be reused rather than retrained. For a fresh experiment, move the previous output directory aside before execution. Keep Paper 1's configured output basenames `paper1_outputs_v2` (original) and `paper1_outputs_lr_floor` (follow-up) unchanged.

Paper 1 and Paper 2 already have full training enabled in the supplied notebooks. Paper 3 requires the environment flags above; without them it defaults to smoke mode with training and analysis disabled.

## Outputs

| Experiment | Figures, metrics and checkpoints |
|---|---|
| Paper 1 original | `paper1_outputs_v2/full/` |
| Paper 1 follow-up | `paper1_outputs_lr_floor/full/` |
| Paper 2 | `paper2_figure3_outputs_v2/reproduction/` |
| Paper 3 | `paper3_figure7_outputs_v2/full/` |

Successful reruns save native cell outputs in `executed/*_rerun.ipynb`. With this `nbconvert` command, the rerun notebook is written after execution finishes; training checkpoints are saved separately during training. Error messages are in `logs/`. The committed notebooks can be opened to inspect their saved results without retraining.

Datasets, packed caches and large checkpoints are not required to be committed. Keep the executed notebook outputs when committing all four notebooks.

## Papers

- Zhang et al., [Understanding Deep Learning Requires Rethinking Generalization](https://openreview.net/forum?id=Sy8gdB9xx).
- Keskar et al., [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://openreview.net/forum?id=H1oyRlYgg).
- Higgins et al., [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl).
