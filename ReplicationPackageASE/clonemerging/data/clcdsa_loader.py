"""CLCDSA dataset construction utilities.

Parses the CLCDSA source solutions and assembles balanced cross-language clone
and non-clone pairs used to build the training and evaluation sets.
"""

import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from data.sketch_extractor import Sketch
except Exception:
    Sketch = Any


@dataclass
class ClonePair:
    code_a: str
    code_b: str
    label: int
    sketch_a: Optional[Sketch] = None
    sketch_b: Optional[Sketch] = None
    lang_a: str = "java"
    lang_b: str = "python"


@dataclass
class CLCDSADataset:
    train: List[ClonePair]
    valid: List[ClonePair]
    test: List[ClonePair]

    @staticmethod
    def load(
        data_dir: str = "data/CLCDSA/Source Codes/",
        cache_path: str = "data/CLCDSA/clcdsa_cache.pkl",
    ) -> "CLCDSADataset":
        if os.path.exists(cache_path):
            print(f"[clcdsa_loader] Loading from cache: {cache_path}")
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        print("[clcdsa_loader] Cache not found - building from source files ...")

        problem_files = _scan_problem_files(data_dir)
        problem_ids = sorted(problem_files.keys())

        if not problem_ids:
            dataset = CLCDSADataset(train=[], valid=[], test=[])
            _write_cache(dataset, cache_path)
            return dataset

        rng = random.Random(42)
        rng.shuffle(problem_ids)

        n_total = len(problem_ids)
        n_train = int(n_total * 0.70)
        n_valid = int(n_total * 0.15)
        n_test = n_total - n_train - n_valid

        train_ids = problem_ids[:n_train]
        valid_ids = problem_ids[n_train : n_train + n_valid]
        test_ids = problem_ids[n_train + n_valid : n_train + n_valid + n_test]

        print(
            f"[clcdsa_loader] Problems - total: {n_total}, "
            f"train: {len(train_ids)}, valid: {len(valid_ids)}, test: {len(test_ids)}"
        )

        train_pairs = _build_pairs_for_split(train_ids, problem_files, rng)
        valid_pairs = _build_pairs_for_split(valid_ids, problem_files, rng)
        test_pairs = _build_pairs_for_split(test_ids, problem_files, rng)

        dataset = CLCDSADataset(train=train_pairs, valid=valid_pairs, test=test_pairs)
        _write_cache(dataset, cache_path)

        print(
            f"[clcdsa_loader] Pairs - train: {len(train_pairs):,}, "
            f"valid: {len(valid_pairs):,}, test: {len(test_pairs):,}"
        )
        return dataset


def _write_cache(dataset: CLCDSADataset, cache_path: str) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[clcdsa_loader] Dataset cached to {cache_path}")


def _scan_problem_files(data_dir: str) -> Dict[str, Dict[str, List[str]]]:
    sources = ["AtCoder", "CodeJamData"]
    problem_files: Dict[str, Dict[str, List[str]]] = {}

    for source in sources:
        source_root = os.path.join(data_dir, source)
        if not os.path.isdir(source_root):
            continue

        for root, _dirs, files in os.walk(source_root):
            java_files = [os.path.join(root, name) for name in files if name.endswith(".java")]
            python_files = [os.path.join(root, name) for name in files if name.endswith(".py")]

            if not java_files or not python_files:
                continue

            problem_id = os.path.relpath(root, data_dir)
            problem_files[problem_id] = {
                "java": sorted(java_files),
                "python": sorted(python_files),
            }

    print(f"[clcdsa_loader] Eligible problems with both langs: {len(problem_files):,}")
    return problem_files


def _read_utf8(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except (UnicodeDecodeError, OSError):
        return None


def _build_pairs_for_split(
    split_problem_ids: List[str],
    problem_files: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
) -> List[ClonePair]:
    pairs: List[ClonePair] = []
    code_cache: Dict[str, Optional[str]] = {}

    python_by_problem = {
        pid: problem_files[pid]["python"]
        for pid in split_problem_ids
        if problem_files[pid]["python"]
    }

    for problem_id in split_problem_ids:
        java_paths = problem_files[problem_id]["java"]
        py_paths = problem_files[problem_id]["python"]

        if not java_paths or not py_paths:
            continue

        negative_problem_candidates = [
            pid for pid in split_problem_ids if pid != problem_id and pid in python_by_problem
        ]
        if not negative_problem_candidates:
            continue

        for java_path in java_paths:
            if java_path not in code_cache:
                code_cache[java_path] = _read_utf8(java_path)
            code_java = code_cache[java_path]
            if code_java is None:
                continue

            for py_path in py_paths:
                if py_path not in code_cache:
                    code_cache[py_path] = _read_utf8(py_path)
                code_py = code_cache[py_path]
                if code_py is None:
                    continue

                pairs.append(ClonePair(code_a=code_java, code_b=code_py, label=1))

                neg_problem_id = rng.choice(negative_problem_candidates)
                neg_py_path = rng.choice(python_by_problem[neg_problem_id])

                if neg_py_path not in code_cache:
                    code_cache[neg_py_path] = _read_utf8(neg_py_path)
                code_neg_py = code_cache[neg_py_path]

                if code_neg_py is not None:
                    pairs.append(ClonePair(code_a=code_java, code_b=code_neg_py, label=0))

    rng.shuffle(pairs)
    return pairs
