#!/bin/bash -l
#SBATCH --job-name=ethos_tokenize::proj=IRB2023P002279,
#SBATCH --time=6:00:00
#SBATCH --partition=defq
#SBATCH --output=ethos_tokenize.log

# this script is intended to be run from the project root
suffix=$1
if [[ -n "$suffix" && ! "$suffix" =~ ^[-_] ]]; then
    suffix="-$suffix"
fi

input_dir="data/mimic-2.2-meds${suffix//_/-}/data"
output_dir="data/tokenized_datasets/mimic${suffix//-/_}"

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH

# Install ethos
cd /ethos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null
"

script_body="
set -e

clear
ethos_tokenize -m worker='range(0,7)' \
    input_dir=$input_dir/train \
    output_dir=$output_dir \
    out_fn=train

ethos_tokenize -m worker='range(0,2)' \
    input_dir=$input_dir/test \
    vocab=$output_dir/train \
    output_dir=$output_dir \
    out_fn=test
"

module load singularity 2>/dev/null

if command -v singularity >/dev/null; then
    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -c "${singularity_preamble}${script_body}"
else
    echo Singularity not found, running using locally using bash.
    bash -c "${script_body}"
fi
