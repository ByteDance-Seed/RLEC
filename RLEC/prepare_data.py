# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the mixed DAPO-Math/HotpotQA dataset used by RLEC."""

import json
import os
import random
from pathlib import Path


RLEC_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RLEC_DATA_ROOT", RLEC_DIR / "data"))
DAPO_JSON = Path(os.environ.get("DAPO_JSON", DATA_ROOT / "DAPO-Math-17k" / "train.json"))
HOTPOT_TRAIN_JSON = Path(
    os.environ.get("HOTPOT_TRAIN_JSON", DATA_ROOT / "HotPotQA" / "train.json")
)
HOTPOT_TEST_JSON = Path(
    os.environ.get("HOTPOT_TEST_JSON", DATA_ROOT / "HotPotQA" / "test.json")
)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-30B-A3B-Base")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", RLEC_DIR / "data" / "processed"))

SOURCE_CONFIG = {
    "dapo-math": {"max_prompt_tokens": 1024, "ability": "math"},
    "hotpot": {"max_prompt_tokens": 4096, "ability": "qa"},
}

TRAIN_PER_SOURCE = 15_000
DAPO_TEST_SIZE = 500
TRAIN_REPEAT = 5
DATA_SEED = 20260806

# Ten percent of the training data uses an ordinary QA objective. The rest
# carries an explicit, uniformly sampled confidence threshold.
RATIO_QA = 0.1
P_MIN_PCT = 0
P_MAX_PCT = 90
EVAL_CONF_PCTS = (10, 50, 90)

IDK_TEXT = "I am not sure."
THINK_INSTRUCTION = (
    "Work through this problem step by step, then honestly assess how "
    "confident you are that your solution is correct."
)
CONF_INSTRUCTION = (
    "Only give a final answer if you are at least {threshold}% confident that "
    "it is correct; otherwise do not guess and answer with "
    "\\boxed{{I am not sure.}} instead."
)
BOX_INSTRUCTION = (
    "Put your final answer inside \\boxed{}. If you are not confident enough, "
    "put \\boxed{I am not sure.} instead."
)
QA_INSTRUCTION = "Work through this problem step by step. Put your final answer inside \\boxed{}."

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return _tokenizer


def _count_tokens(text):
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _read_json_qa(path, source):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []
    for source_index, record in enumerate(data):
        question = record.get("question", record.get("problem"))
        answer = record.get("answer")
        if question is None or answer is None:
            continue
        question = str(question).strip()
        if not question:
            continue
        rows.append(
            {
                "question": question,
                "answer": str(answer),
                "source": source,
                "source_index": source_index,
            }
        )
    return rows


def _confidence_prompt(question, p):
    threshold = round(min(0.99, max(0.01, float(p))) * 100)
    return (
        question.strip()
        + "\n\n"
        + THINK_INSTRUCTION
        + " "
        + CONF_INSTRUCTION.format(threshold=threshold)
        + " "
        + BOX_INSTRUCTION
    )


def _qa_prompt(question):
    return question.strip() + "\n\n" + QA_INSTRUCTION


def _filter_by_length(records, source):
    limit = SOURCE_CONFIG[source]["max_prompt_tokens"]
    kept = []
    for record in records:
        # Use the longest RLEC prompt for a conservative source-specific filter.
        if _count_tokens(_confidence_prompt(record["question"], 0.5)) <= limit:
            kept.append(record)
    print(f"[filter:{source}] kept={len(kept):,}/{len(records):,}, limit={limit}")
    return kept


def _load_and_split_records():
    dapo = _filter_by_length(_read_json_qa(DAPO_JSON, "dapo-math"), "dapo-math")
    hotpot_train = _filter_by_length(
        _read_json_qa(HOTPOT_TRAIN_JSON, "hotpot"), "hotpot"
    )
    hotpot_test = _filter_by_length(
        _read_json_qa(HOTPOT_TEST_JSON, "hotpot"), "hotpot"
    )

    required_dapo = TRAIN_PER_SOURCE + DAPO_TEST_SIZE
    if len(dapo) < required_dapo:
        raise ValueError(f"DAPO only has {len(dapo):,} eligible rows; need {required_dapo:,}")
    if len(hotpot_train) < TRAIN_PER_SOURCE:
        raise ValueError(
            f"HotpotQA only has {len(hotpot_train):,} eligible training rows; "
            f"need {TRAIN_PER_SOURCE:,}"
        )

    random.Random(DATA_SEED).shuffle(dapo)
    random.Random(DATA_SEED + 1).shuffle(hotpot_train)
    train = dapo[:TRAIN_PER_SOURCE] + hotpot_train[:TRAIN_PER_SOURCE]
    test = dapo[TRAIN_PER_SOURCE:required_dapo] + hotpot_test
    random.Random(DATA_SEED + 2).shuffle(train)
    return train, test


def _repeat_train(records):
    repeated = []
    for repeat_index in range(TRAIN_REPEAT):
        current = list(records)
        random.Random(DATA_SEED + 100 + repeat_index).shuffle(current)
        repeated.extend(current)
    return repeated


def _sample(record, index, split, kind, confidence=None, suffix=""):
    source = record["source"]
    prompt = (
        _qa_prompt(record["question"])
        if kind == "qa"
        else _confidence_prompt(record["question"], confidence)
    )
    data_source = source if split == "train" else f"{source}{suffix}"
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": SOURCE_CONFIG[source]["ability"],
        "reward_model": {"style": "rule", "ground_truth": record["answer"]},
        "extra_info": {
            "split": split,
            "index": index,
            "source_index": record["source_index"],
            "source": source,
            "question": record["question"],
            "answer": record["answer"],
            "kind": kind,
            "confidence": confidence,
            "eval_group": data_source if split == "test" else None,
        },
    }


def _build_train(records):
    rng = random.Random(DATA_SEED + 1000)
    qa_count = round(RATIO_QA * len(records))
    kinds = ["qa"] * qa_count + ["conf"] * (len(records) - qa_count)
    rng.shuffle(kinds)

    samples = []
    for index, (record, kind) in enumerate(zip(records, kinds, strict=True)):
        confidence = (
            None
            if kind == "qa"
            else round(rng.uniform(P_MIN_PCT, P_MAX_PCT) / 100, 2)
        )
        samples.append(_sample(record, index, "train", kind, confidence))
    return samples


def _build_test(records):
    samples = [_sample(record, index, "test", "qa", suffix="__qa") for index, record in enumerate(records)]
    for percentage in EVAL_CONF_PCTS:
        confidence = percentage / 100
        samples.extend(
            _sample(
                record,
                index,
                "test",
                "conf",
                confidence,
                suffix=f"__conf{percentage}",
            )
            for index, record in enumerate(records)
        )
    return samples


def main():
    import pandas as pd

    train_records, test_records = _load_and_split_records()
    train_samples = _build_train(_repeat_train(train_records))
    test_samples = _build_test(test_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.parquet"
    test_path = OUTPUT_DIR / "test.parquet"
    pd.DataFrame(train_samples).to_parquet(train_path, index=False)
    pd.DataFrame(test_samples).to_parquet(test_path, index=False)
    print(f"[train] {len(train_samples):,} -> {train_path}")
    print(f"[test]  {len(test_samples):,} -> {test_path}")


if __name__ == "__main__":
    main()
