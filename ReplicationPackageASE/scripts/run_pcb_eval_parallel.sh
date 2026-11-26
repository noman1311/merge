#!/bin/bash
# Parallel evaluation of PCB-Merging checkpoints.
#
# Evaluates 9 PCB merged checkpoints on three datasets:
# BCB, CLCDSA, GPTCloneBench. Total 27 evaluation runs, parallelized
# across 4 GPUs in waves of 4.
#
# Usage:
#   cd /path/to/ReplicationPackageASE
#   conda activate clonemerge
#   unset CUDA_VISIBLE_DEVICES   # use all 4 GPUs
#   tmux new -s pcb_eval
#   bash run_pcb_eval_parallel.sh 2>&1 | tee logs/pcb_eval_parallel.log
#   # Ctrl+B D to detach

set -uo pipefail

SCRIPT="clonemerging/evaluation/evaluate_models.py"
RESULTS_DIR="clonemerging/evaluation/results"
mkdir -p "$RESULTS_DIR" logs

# All 9 PCB checkpoints
CHECKPOINTS=(
    "merged_unixcoder_pcb"
    "merged_unixcoder_pcb_seed2"
    "merged_unixcoder_pcb_seed3"
    "merged_codebert_pcb"
    "merged_codebert_pcb_seed2"
    "merged_codebert_pcb_seed3"
    "merged_crossbase_pcb"
    "merged_crossbase_pcb_seed2"
    "merged_crossbase_pcb_seed3"
)

DATASETS=("bcb" "clcdsa" "gptclonebench")

# Build the full job list: 9 checkpoints x 3 datasets = 27 jobs
JOBS=()
for ckpt in "${CHECKPOINTS[@]}"; do
    for dset in "${DATASETS[@]}"; do
        JOBS+=("$ckpt|$dset")
    done
done

TOTAL=${#JOBS[@]}

echo "=========================================="
echo "Parallel PCB evaluation"
echo "Start: $(date)"
echo "Total jobs: $TOTAL (9 checkpoints x 3 datasets)"
echo "GPUs: 4"
echo "=========================================="

run_eval() {
    local gpu_id="$1"
    local ckpt="$2"
    local dset="$3"
    local label="${ckpt}_${dset}"
    local out_file="$RESULTS_DIR/${label}.json"
    local log_file="logs/eval_${label}.log"

    echo "[$(date '+%H:%M:%S')] Starting [$label] on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES="$gpu_id" python "$SCRIPT" \
        --checkpoint_dir "checkpoints/$ckpt" \
        --dataset_type "$dset" \
        --output_file "$out_file" \
        --batch_size 64 \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Finished [$label] on GPU $gpu_id (ok)"
    else
        echo "[$(date '+%H:%M:%S')] FAILED [$label] on GPU $gpu_id (exit $exit_code) -- see $log_file"
    fi
    return $exit_code
}

# Dispatch jobs in waves of 4
WAVE=1
IDX=0
TOTAL_FAILED=0

while [ $IDX -lt $TOTAL ]; do
    echo ""
    echo "Wave $WAVE: launching up to 4 jobs (jobs $((IDX+1))-$((IDX+4)) of $TOTAL)"
    echo ""

    PIDS=()
    LABELS=()
    for gpu_id in 0 1 2 3; do
        if [ $IDX -ge $TOTAL ]; then
            break
        fi
        IFS='|' read -r ckpt dset <<< "${JOBS[$IDX]}"
        run_eval "$gpu_id" "$ckpt" "$dset" &
        PIDS+=($!)
        LABELS+=("${ckpt}_${dset}")
        IDX=$((IDX+1))
    done

    echo "Wave $WAVE PIDs: ${PIDS[*]}"
    echo "Waiting for wave $WAVE..."

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  [wave $WAVE] ${LABELS[$i]} -> FAILED (exit $rc)"
            TOTAL_FAILED=$((TOTAL_FAILED+1))
        fi
    done

    echo "Wave $WAVE done."
    WAVE=$((WAVE+1))
done

echo ""
echo "=========================================="
echo "All PCB evaluations complete"
echo "End: $(date)"
echo "Failed jobs: $TOTAL_FAILED / $TOTAL"
echo "=========================================="
echo ""
echo "Result files in $RESULTS_DIR:"
ls -lh "$RESULTS_DIR"/merged_*_pcb*.json "$RESULTS_DIR"/merged_crossbase_pcb*.json 2>/dev/null | tail -30