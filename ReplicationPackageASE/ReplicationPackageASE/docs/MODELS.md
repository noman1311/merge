# Models and checkpoints

The fine-tuned specialists and merged checkpoints are large and are not
redistributed in this package. They are reproduced from the training and merging
scripts. This document lists the checkpoints the experiments expect, and how to
produce each group. All checkpoints are written under `checkpoints/` at the
package root, which is the directory the evaluation and merging scripts read
from.

## Specialists

Each specialist is a base encoder fine-tuned on one benchmark with
`clonemerging/training/finetune.py`.

| Checkpoint | Base model | Benchmark |
|------------|-----------|-----------|
| `codebert_bcb` | microsoft/codebert-base | BigCloneBench |
| `codebert_clcdsa` | microsoft/codebert-base | CLCDSA |
| `graphcodebert_bcb` | microsoft/graphcodebert-base | BigCloneBench |
| `graphcodebert_clcdsa` | microsoft/graphcodebert-base | CLCDSA |
| `unixcoder_bcb` | microsoft/unixcoder-base | BigCloneBench |
| `unixcoder_clcdsa` | microsoft/unixcoder-base | CLCDSA |

Example:

```bash
python clonemerging/training/finetune.py \
    --model_name microsoft/codebert-base \
    --dataset_type bcb \
    --output_dir checkpoints/codebert_bcb \
    --epochs 5 --batch_size 32 --lr 2e-5 --seed 42
```

### Seed variants

Multi-seed reporting uses additional specialists trained with seeds 2 and 3
(for example `codebert_clcdsa_seed2`, `unixcoder_bcb_seed3`). Reproduce them by
repeating the command above with `--seed 2` or `--seed 3` and the matching
output directory name.

### Multi-task baseline

`unixcoder_combined` is the "train on all data" reference, produced with
`clonemerging/training/finetune_combined.py` on BigCloneBench and CLCDSA together.

## Merged checkpoints

All merges operate on the specialists above and require no training data.

| Group | Method | Script |
|-------|--------|--------|
| `merged_equal`, `merged_ties_only`, `merged_dare_ties`, `merged_*_heavy` | Weighted averaging / TIES / DARE-TIES | `clonemerging/merging/parameter_merging.py` |
| `merged_codebert_wudi`, `merged_unixcoder_wudi`, `merged_crossbase_wudi` (with seed variants) | WUDI-Merging | `clonemerging/merging/wudi_merging.py` |
| `merged_codebert_pcb`, `merged_unixcoder_pcb`, `merged_crossbase_pcb` (with seed variants) | PCB-Merging | `clonemerging/merging/pcb_merging.py` |
| `franken_greedy_v3` (CB+GCB), `franken_unixcoder` (UX+UX) | Architecture (layer-composition) merging | `clonemerging/merging/architecture_merging.py` |

The runner scripts in `scripts/` reproduce the full WUDI and PCB merge sets and
their evaluation across seeds and GPUs. Each merging script accepts `--help` for
the complete option list; the merge weights, drop rate, trim percentile, WUDI
step count, PCB drop rate and intra-balancing scale, and seed used for each
configuration are set there.

## Evaluation

Every checkpoint is evaluated with `clonemerging/evaluation/evaluate_models.py` on
`bcb`, `clcdsa`, and `gptclonebench`. The resulting JSON files are provided under
`clonemerging/evaluation/results/`, so the reported numbers can be inspected
without re-running training, merging, or inference.
