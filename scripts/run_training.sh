#!/bin/bash -l
#SBATCH --job-name=ethos_train::proj=IRB2023P002279,
#SBATCH --time=3-0
#SBATCH --partition=defq
#SBATCH --gres=gpu:8
#SBATCH --output=ethos_train.log

# this script is intended to be run from the project root
export OMP_NUM_THREADS=20

dataset="mimic_ed"
dataset_name="mimic"

data_path=data/tokenized_datasets/$dataset
clear
if [[ ! -d $data_path ]]; then
    echo "Dataset directory not found: $data_path"
    exit 1
fi

shift 1

BATCH_SIZE=32
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001

model_name="layer_${N_LAYER}_do_${DROPOUT}"

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/:/.singularity.d/libs/

# Install ethos
cd /ethos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null

# Use other tmp dir to avoid /tmp filling up and preserve the cache across the runs
export TORCHINDUCTOR_CACHE_DIR=/ethos/torchinductor_cache
"

script_body="
torchrun --no_python --standalone --nproc_per_node=\${NUM_GPUS} ethos_train \
  data_fp=$data_path/train \
  val_size=6 \
  batch_size=$BATCH_SIZE \
  n_positions=$N_POSITIONS \
  n_layer=$N_LAYER \
  n_head=$N_HEAD \
  n_embd=$N_EMBD \
  dropout=$DROPOUT \
  lr=$LR \
  min_lr=$MIN_LR \
  log_interval=10 \
  eval_interval=1500 \
  gradient_accumulation_steps=16 \
  warmup_iters=5000 \
  max_iters=200000 \
  lr_decay_iters=100000 \
  wandb_log=true \
  wandb_project="ethos-meds-$dataset_name" \
  wandb_run_name=$model_name \
  $* \
  out_dir="${data_path}/models/${model_name}"
"

module load singularity 2>/dev/null

if command -v singularity >/dev/null; then
    export NUM_GPUS=${SLURM_GPUS_ON_NODE}
    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -c "${singularity_preamble}${script_body}"
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    export NUM_GPUS
    bash -c "${script_body}"
fi
