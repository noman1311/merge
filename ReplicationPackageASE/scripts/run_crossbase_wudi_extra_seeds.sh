#!/bin/bash
# Two additional cross-base WUDI seeds for consistency with the three-seed reporting
# elsewhere. Merges then evaluates on BCB, CLCDSA, GPTCloneBench.
#
# Usage:
#   cd /path/to/ReplicationPackageASE
#   conda activate clonemerge
#   unset CUDA_VISIBLE_DEVICES
#   nohup bash run_crossbase_wudi_extra_seeds.sh > logs/crossbase_wudi_extra.log 2>&1 &

set -uo pipefail

MERGE_SCRIPT="clonemerging/merging/wudi_merging.py"
EVAL_SCRIPT="clonemerging/evaluation/evaluate_models.py"
ROOT="checkpoints"
RESULTS_DIR="clonemerging/evaluation/results"
NUM_STEPS=300
LR=1e-5
MERGE_SEED=42

mkdir -p logs "$RESULTS_DIR"

echo "=========================================="
echo "Cross-base WUDI extra seeds"
echo "Start: $(date)"
echo "=========================================="

# Phase 1: Two merges in parallel (one per GPU)
echo ""
echo "Phase 1: cross-base WUDI merges (seed2, seed3)"
echo ""

CUDA_VISIBLE_DEVICES=0 python "$MERGE_SCRIPT" \
    --base_model microsoft/unixcoder-base \
    --checkpoints unixcoder_bcb_seed2 codebert_clcdsa_seed2 \
    --checkpoint_root "$ROOT" \
    --base_models microsoft/unixcoder-base microsoft/codebert-base \
    --weights 1.0 1.0 \
    --num_steps $NUM_STEPS --lr $LR \
    --seed $MERGE_SEED \
    --output_dir "$ROOT/merged_crossbase_wudi_seed2" \
    > logs/wudi_crossbase_seed2.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python "$MERGE_SCRIPT" \
    --base_model microsoft/unixcoder-base \
    --checkpoints unixcoder_bcb_seed3 codebert_clcdsa_seed3 \
    --checkpoint_root "$ROOT" \
    --base_models microsoft/unixcoder-base microsoft/codebert-base \
    --weights 1.0 1.0 \
    --num_steps $NUM_STEPS --lr $LR \
    --seed $MERGE_SEED \
    --output_dir "$ROOT/merged_crossbase_wudi_seed3" \
    > logs/wudi_crossbase_seed3.log 2>&1 &
PID2=$!

echo "Merge PIDs: $PID1 $PID2"
echo "Waiting for merges..."
wait $PID1; M1=$?
wait $PID2; M2=$?
echo "Merge results: seed2=$M1, seed3=$M2"

if [ $M1 -ne 0 ] || [ $M2 -ne 0 ]; then
    echo "ABORTING: merge failure detected; see logs/wudi_crossbase_seed{2,3}.log"
    exit 1
fi

# Phase 2: Six evaluations in two waves of 3 + 3 across GPUs 0..2
echo ""
echo "Phase 2: evaluations (6 jobs across 2 waves)"

run_eval() {
    local gpu_id="$1"
    local ckpt="$2"
    local dset="$3"
    local label="${ckpt}_${dset}"
    echo "[$(date '+%H:%M:%S')] Starting eval [$label] on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" python "$EVAL_SCRIPT" \
        --checkpoint_dir "checkpoints/$ckpt" \
        --dataset_type "$dset" \
        --output_file "$RESULTS_DIR/${label}.json" \
        --batch_size 64 \
        > "logs/eval_${label}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Done eval [$label] on GPU $gpu_id"
    else
        echo "[$(date '+%H:%M:%S')] FAILED eval [$label] on GPU $gpu_id (exit $rc)"
    fi
    return $rc
}

echo ""
echo "Wave 1: seed2 x {bcb, clcdsa, gptclonebench}"
run_eval 0 "merged_crossbase_wudi_seed2" "bcb" &
PA=$!
run_eval 1 "merged_crossbase_wudi_seed2" "clcdsa" &
PB=$!
run_eval 2 "merged_crossbase_wudi_seed2" "gptclonebench" &
PC=$!
wait $PA $PB $PC

echo ""
echo "Wave 2: seed3 x {bcb, clcdsa, gptclonebench}"
run_eval 0 "merged_crossbase_wudi_seed3" "bcb" &
PA=$!
run_eval 1 "merged_crossbase_wudi_seed3" "clcdsa" &
PB=$!
run_eval 2 "merged_crossbase_wudi_seed3" "gptclonebench" &
PC=$!
wait $PA $PB $PC

echo ""
echo "=========================================="
echo "Cross-base WUDI extra seeds complete"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "New result files:"
ls -lh "$RESULTS_DIR"/merged_crossbase_wudi_seed*.json 2>/dev/null