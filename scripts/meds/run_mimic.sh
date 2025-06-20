#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MIMICIV_RAW_DIR> <MIMICIV_PRE_MEDS_DIR> <MIMICIV_MEDS_COHORT_DIR>"
    echo
    echo "This script processes MIMIC-IV data through several steps, handling raw data conversion,"
    echo "sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  MIMICIV_RAW_DIR                                Directory containing raw MIMIC-IV data files."
    echo "  MIMICIV_PREMEDS_DIR                            Output directory for pre-MEDS data."
    echo "  MIMICIV_MEDS_DIR                               Output directory for processed MEDS data."
    echo "  (OPTIONAL) MIMIC extension                     Choose the extension to parse, available are '' or 'ed'."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

echo "Unsetting SLURM_CPU_BIND in case you're running this on a slurm interactive node with slurm parallelism"
unset SLURM_CPU_BIND

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export MIMICIV_RAW_DIR=$1
export MIMICIV_PRE_MEDS_DIR=$2
export MIMICIV_MEDS_COHORT_DIR=$3
shift 3

case $1 in
ed | cxr)
    echo "Using extension: $1"
    extension_suffix="-$1"
    ;;
"")
    extension_suffix=""
    ;;
*)
    echo "Error: Invalid extension provided. Choose from 'ed' or 'cxr'."
    display_help
    ;;
esac

EVENT_CONVERSION_CONFIG_FP="$(pwd)/mimic/configs/event_configs${extension_suffix}.yaml"
# We export these variables separately from their assignment so that any errors during assignment are caught.
export EVENT_CONVERSION_CONFIG_FP

echo "Running pre-MEDS conversion."
PRE_MEDS_PY_FP="$(pwd)/mimic/pre_MEDS.py"
python "$PRE_MEDS_PY_FP" input_dir="$MIMICIV_RAW_DIR" cohort_dir="$MIMICIV_PRE_MEDS_DIR"

if [ -z "$N_WORKERS" ]; then
    echo "Setting N_WORKERS to 1 to avoid issues with the runners."
    export N_WORKERS="1"
fi

echo "Running extraction pipeline."
PIPELINE_CONFIG_FP="$(pwd)/mimic/configs/extract_MIMIC.yaml"
MEDS_transform-pipeline "pipeline_config_fp=$PIPELINE_CONFIG_FP"
