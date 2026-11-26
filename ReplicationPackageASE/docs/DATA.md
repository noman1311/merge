# Datasets

The study uses three public clone-detection benchmarks. They are not
redistributed in this package; download each from its original source and
arrange it under a top-level `data/` directory as shown below. The dataset
loaders in `clonemerging/data/` expect these paths by default, and each loader
also accepts an explicit directory argument if you prefer a different location.

## Sources

| Dataset | Used for | Source |
|---------|----------|--------|
| BigCloneBench (BCB) | Java clone detection (train and test) | CodeXGLUE, https://github.com/microsoft/CodeXGLUE (Code-Code / Clone-detection-BigCloneBench) |
| CLCDSA | Cross-language clone detection (train and test) | https://github.com/Kawser-nerd/CLCDSA |
| GPTCloneBench | Semantic clone detection (test, and the LLM baseline) | Zenodo, https://doi.org/10.5281/zenodo.10198952 (CC-BY-4.0) |

CLCDSA provides the cross-language clone data (Java, C#, Python); it is used both
for the cross-language specialist and as a cross-language test set.

## Expected layout

```
data/
|-- CodeXGLUE/
|   `-- Code-Code/
|       `-- Clone-detection-BigCloneBench/
|           `-- dataset/                     # BigCloneBench train/valid/test
|-- CLCDSA/
|   |-- Source Codes/                        # CLCDSA source files
|   `-- clcdsa_cache.pkl                     # built on first load (see below)
`-- GPTCloneBench/
    `-- GPTCloneBench/
        `-- standalone/                      # extracted semantic clone pairs
```

## Notes

- **BigCloneBench** ships with CodeXGLUE as index files plus a source archive.
  Follow the CodeXGLUE clone-detection instructions to materialise the
  `dataset/` directory.
- **CLCDSA** builds a cache (`clcdsa_cache.pkl`) the first time the loader runs;
  this speeds up subsequent loads and is safe to delete.
- **GPTCloneBench** is distributed as a single archive on Zenodo. Download and
  extract `GPTCloneBench_semantic_clone_pairs.zip`, then point the loader at the
  `standalone/` directory.
- Each benchmark retains its original license. Review and comply with the terms
  at the sources above before use.

## Pre-trained base models

The specialists are fine-tuned from public HuggingFace checkpoints, downloaded
automatically on first use:

- `microsoft/codebert-base`
- `microsoft/graphcodebert-base`
- `microsoft/unixcoder-base`

The large-language-model baselines use `Qwen/Qwen2.5-Coder-7B-Instruct` and
`deepseek-ai/deepseek-coder-6.7b-instruct`.
