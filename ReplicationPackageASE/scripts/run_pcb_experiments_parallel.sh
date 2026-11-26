#!/bin/bash
# Parallel PCB-Merging (NeurIPS 2024) experiments.
#
# Runs 9 merges (6 same-base + 3 cross-base) in waves of up to 4 across GPUs 0-3.
# PCB is closed-form (intra-balance, inter-balance, drop, rescale) and runs in
# seconds to a minute per merge.
#
# Seeds in the directory names refer to the specialist training seed. The merge
# itself is deterministic and always uses --seed 42.
#
# Usage:
#   cd /path/to/ReplicationPackageASE
#   tmux new -s pcb
#   conda activate clonemerge
#   bash run_pcb_experiments_parallel.sh 2>&1 | tee logs/pcb_parallel.log
#   # Ctrl+B D to detach

set -uo pipefail

SCRIPT="clonemerging/merging/pcb_merging.py"
ROOT="checkpoints"
DROP_RATE=0.1
INTRA_SCALE=1.0
MERGE_SEED=42

mkdir -p logs

# Helper function: run a single PCB job pinned to a specific GPU.
# Args: $1 = GPU id, $2 = job label, $3 = base_model, $4 = checkpoint1,
#       $5 = checkpoint2, $6 = base1, $7 = base2, $8 = output_dir
run_pcb() {
    local gpu_id="$1"
    local label="$2"
    local base_model="$3"
    local ckpt1="$4"
    local ckpt2="$5"
    local base1="$6"
    local base2="$7"
    local out_dir="$8"
    local log_file="logs/pcb_${label}.log"

    echo "[$(date '+%H:%M:%S')] Starting [$label] on GPU $gpu_id -> $out_dir"

    CUDA_VISIBLE_DEVICES="$gpu_id" python "$SCRIPT" \
        --base_model "$base_model" \
        --checkpoints "$ckpt1" "$ckpt2" \
        --checkpoint_root "$ROOT" \
        --base_models "$base1" "$base2" \
        --weights 1.0 1.0 \
        --drop_rate $DROP_RATE \
        --intra_scale $INTRA_SCALE \
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
echo "Parallel PCB-Merging experiments (NeurIPS 2024)"
echo "Start: $(date)"
echo "drop_rate:    $DROP_RATE"
echo "intra_scale:  $INTRA_SCALE"
echo "Merge seed:   $MERGE_SEED"
echo "GPUs:         4"
echo "=========================================="

# -----------------------------------------------------------------------------
# Wave 1: 4 jobs in parallel (one per GPU)
#   UX same-base (3 seeds) + CB same-base seed 42
# -----------------------------------------------------------------------------
echo ""
echo "Wave 1/3: launching 4 jobs (UX seeds 42/2/3, CB seed 42)"
echo ""

run_pcb 0 "ux_seed42" \
    microsoft/unixcoder-base \
    unixcoder_bcb unixcoder_clcdsa \
    microsoft/unixcoder-base microsoft/unixcoder-base \
    "$ROOT/merged_unixcoder_pcb" &
PID1=$!

run_pcb 1 "ux_seed2" \
    microsoft/unixcoder-base \
    unixcoder_bcb_seed2 unixcoder_clcdsa_seed2 \
    microsoft/unixcoder-base microsoft/unixcoder-base \
    "$ROOT/merged_unixcoder_pcb_seed2" &
PID2=$!

run_pcb 2 "ux_seed3" \
    microsoft/unixcoder-base \
    unixcoder_bcb_seed3 unixcoder_clcdsa_seed3 \
    microsoft/unixcoder-base microsoft/unixcoder-base \
    "$ROOT/merged_unixcoder_pcb_seed3" &
PID3=$!

run_pcb 3 "cb_seed42" \
    microsoft/codebert-base \
    codebert_bcb codebert_clcdsa \
    microsoft/codebert-base microsoft/codebert-base \
    "$ROOT/merged_codebert_pcb" &
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
# Wave 2: 3 jobs in parallel
#   CB same-base seeds 2/3 + cross-base seed 42
# -----------------------------------------------------------------------------
echo ""
echo "Wave 2/3: launching 3 jobs (CB seed 2/3, crossbase seed 42)"
echo ""

run_pcb 0 "cb_seed2" \
    microsoft/codebert-base \
    codebert_bcb codebert_clcdsa_seed2 \
    microsoft/codebert-base microsoft/codebert-base \
    "$ROOT/merged_codebert_pcb_seed2" &
PID1=$!

run_pcb 1 "cb_seed3" \
    microsoft/codebert-base \
    codebert_bcb codebert_clcdsa_seed3 \
    microsoft/codebert-base microsoft/codebert-base \
    "$ROOT/merged_codebert_pcb_seed3" &
PID2=$!

run_pcb 2 "crossbase" \
    microsoft/unixcoder-base \
    unixcoder_bcb codebert_clcdsa \
    microsoft/unixcoder-base microsoft/codebert-base \
    "$ROOT/merged_crossbase_pcb" &
PID3=$!

echo "Wave 2 PIDs: $PID1 $PID2 $PID3"
echo "Waiting for Wave 2 to complete..."

wait $PID1; W2_S1=$?
wait $PID2; W2_S2=$?
wait $PID3; W2_S3=$?

echo ""
echo "Wave 2 results: cb_seed2=$W2_S1, cb_seed3=$W2_S2, crossbase=$W2_S3"

# -----------------------------------------------------------------------------
# Wave 3: 2 jobs in parallel
#   cross-base seeds 2/3
# -----------------------------------------------------------------------------
echo ""
echo "Wave 3/3: launching 2 jobs (crossbase seeds 2/3)"
echo ""

run_pcb 0 "crossbase_seed2" \
    microsoft/unixcoder-base \
    unixcoder_bcb_seed2 codebert_clcdsa_seed2 \
    microsoft/unixcoder-base microsoft/codebert-base \
    "$ROOT/merged_crossbase_pcb_seed2" &
PID1=$!

run_pcb 1 "crossbase_seed3" \
    microsoft/unixcoder-base \
    unixcoder_bcb_seed3 codebert_clcdsa_seed3 \
    microsoft/unixcoder-base microsoft/codebert-base \
    "$ROOT/merged_crossbase_pcb_seed3" &
PID2=$!

echo "Wave 3 PIDs: $PID1 $PID2"
echo "Waiting for Wave 3 to complete..."

wait $PID1; W3_S1=$?
wait $PID2; W3_S2=$?

echo ""
echo "Wave 3 results: crossbase_seed2=$W3_S1, crossbase_seed3=$W3_S2"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "All PCB experiments complete"
echo "End: $(date)"
echo "=========================================="
echo ""

TOTAL_FAILED=$((W1_S1 + W1_S2 + W1_S3 + W1_S4 + W2_S1 + W2_S2 + W2_S3 + W3_S1 + W3_S2))
if [ $TOTAL_FAILED -ne 0 ]; then
    echo "WARNING: Some runs failed. Check logs/pcb_*.log for details."
fi

echo "Merged checkpoints saved to:"
ls -ld "$ROOT"/merged_*_pcb* "$ROOT"/merged_crossbase_pcb* 2>/dev/null

echo ""
echo "Per-job log files:"
ls -lh logs/pcb_*.log 2>/dev/null

echo ""
echo "Next step: bash run_pcb_eval_parallel.sh"