#!/bin/bash
# Parallel WUDI-Merging experiments.
#
# Launches one WUDI run per GPU (4 in parallel), waits for all to finish, then
# launches the remaining runs the same way.
#
# Seeds in the directory names refer to the specialist training seed. The merge
# itself is deterministic and always uses --seed 42.
#
# Usage:
#   cd /path/to/ReplicationPackageASE
#   tmux new -s wudi
#   conda activate clonemerge
#   bash run_wudi_experiments_parallel.sh 2>&1 | tee logs/wudi_parallel.log
#   # Ctrl+B D to detach

set -uo pipefail  # note: no -e since we want to capture per-job exit codes ourselves

SCRIPT="clonemerging/merging/wudi_merging.py"
ROOT="checkpoints"
NUM_STEPS=300
LR=1e-5
MERGE_SEED=42

mkdir -p logs

# Helper function: run a single WUDI job pinned to a specific GPU.
# Args: $1 = GPU id, $2 = job label, $3 = base_model, $4 = checkpoint1,
#       $5 = checkpoint2, $6 = base1, $7 = base2, $8 = output_dir
run_wudi() {
    local gpu_id="$1"
    local label="$2"
    local base_model="$3"
    local ckpt1="$4"
    local ckpt2="$5"
    local base1="$6"
    local base2="$7"
    local out_dir="$8"
    local log_file="logs/wudi_${label}.log"

    echo "[$(date '+%H:%M:%S')] Starting [$label] on GPU $gpu_id -> $out_dir"

    CUDA_VISIBLE_DEVICES="$gpu_id" python "$SCRIPT" \
        --base_model "$base_model" \
        --checkpoints "$ckpt1" "$ckpt2" \
        --checkpoint_root "$ROOT" \
        --base_models "$base1" "$base2" \
        --weights 1.0 1.0 \
        --num_steps $NUM_STEPS --lr $LR \
        --seed $MERGE_SEED \
        --output_dir "$out_dir" > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Finished [$label] on GPU $gpu_id (success)"
    else
        echo "[$(date '+%H:%M:%S')] FAILED [$label] on GPU $gpu_id (exit $exit_code) - see $log_file"
    fi
    return $exit_code
}

echo "=========================================="
echo "Parallel WUDI-Merging experiments"
echo "Start: $(date)"
echo "Steps per layer: $NUM_STEPS"
echo "Learning rate:   $LR"
echo "Merge seed:      $MERGE_SEED"
echo "GPUs available:  4"
echo "=========================================="

# -----------------------------------------------------------------------------
# Wave 1: 4 jobs in parallel (one per GPU)
# -----------------------------------------------------------------------------
echo ""
echo "Wave 1/2: launching 4 jobs (one per GPU)"
echo ""

run_wudi 0 "ux_seed42" \
    microsoft/unixcoder-base \
    unixcoder_bcb unixcoder_clcdsa \
    microsoft/unixcoder-base microsoft/unixcoder-base \
    "$ROOT/merged_unixcoder_wudi" &
PID1=$!

run_wudi 1 "ux_seed2" \
    microsoft/unixcoder-base \
    unixcoder_bcb_seed2 unixcoder_clcdsa_seed2 \
    microsoft/unixcoder-base microsoft/unixcoder-base \
    "$ROOT/merged_unixcoder_wudi_seed2" &
PID2=$!

run_wudi 2 "ux_seed3" \
    microsoft/unixcoder-base \
    unixcoder_bcb_seed3 unixcoder_clcdsa_seed3 \
    microsoft/unixcoder-base microsoft/unixcoder-base \
    "$ROOT/merged_unixcoder_wudi_seed3" &
PID3=$!

run_wudi 3 "cb_seed42" \
    microsoft/codebert-base \
    codebert_bcb codebert_clcdsa \
    microsoft/codebert-base microsoft/codebert-base \
    "$ROOT/merged_codebert_wudi" &
PID4=$!

echo "Wave 1 PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for Wave 1 to complete..."

wait $PID1; W1_S1=$?
wait $PID2; W1_S2=$?
wait $PID3; W1_S3=$?
wait $PID4; W1_S4=$?

echo ""
echo "Wave 1 results: ux_seed42=$W1_S1, ux_seed2=$W1_S2, ux_seed3=$W1_S3, cb_seed42=$W1_S4"

# -----------------------------------------------------------------------------
# Wave 2: 3 jobs in parallel (3 GPUs, last GPU idle)
# -----------------------------------------------------------------------------
echo ""
echo "Wave 2/2: launching 3 jobs"
echo ""

run_wudi 0 "cb_seed2" \
    microsoft/codebert-base \
    codebert_bcb codebert_clcdsa_seed2 \
    microsoft/codebert-base microsoft/codebert-base \
    "$ROOT/merged_codebert_wudi_seed2" &
PID1=$!

run_wudi 1 "cb_seed3" \
    microsoft/codebert-base \
    codebert_bcb codebert_clcdsa_seed3 \
    microsoft/codebert-base microsoft/codebert-base \
    "$ROOT/merged_codebert_wudi_seed3" &
PID2=$!

run_wudi 2 "crossbase" \
    microsoft/unixcoder-base \
    unixcoder_bcb codebert_clcdsa \
    microsoft/unixcoder-base microsoft/codebert-base \
    "$ROOT/merged_crossbase_wudi" &
PID3=$!

echo "Wave 2 PIDs: $PID1 $PID2 $PID3"
echo "Waiting for Wave 2 to complete..."

wait $PID1; W2_S1=$?
wait $PID2; W2_S2=$?
wait $PID3; W2_S3=$?

echo ""
echo "Wave 2 results: cb_seed2=$W2_S1, cb_seed3=$W2_S2, crossbase=$W2_S3"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "All WUDI experiments complete"
echo "End: $(date)"
echo "=========================================="
echo ""

TOTAL_FAILED=$((W1_S1 + W1_S2 + W1_S3 + W1_S4 + W2_S1 + W2_S2 + W2_S3))
if [ $TOTAL_FAILED -ne 0 ]; then
    echo "WARNING: Some runs failed. Check logs/wudi_*.log for details."
fi

echo "Merged checkpoints saved to:"
ls -ld "$ROOT"/merged_*_wudi* "$ROOT"/merged_crossbase_wudi 2>/dev/null

echo ""
echo "Per-job log files:"
ls -lh logs/wudi_*.log 2>/dev/null