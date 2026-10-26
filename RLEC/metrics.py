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

"""Grade raw RLEC generations and compute accuracy, AOC, and reward score.

Inputs are one or more ``raw.jsonl`` files produced by ``evaluate.py``. Files
from different datasets in the same model run may be supplied together. This
script writes one JSON report and a per-dataset/per-confidence CSV table.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


# ========================== Configuration ==========================
RAW_INPUTS = (
    # "",
)
OUTPUT_PATH = os.environ.get("RLEC_METRICS_OUTPUT", "")

# AOC uses attempted accuracy by default: correct / (correct + incorrect).
# Set to "overall" to use correct / (correct + incorrect + abstain).
AOC_ACC_MODE = "attempted"

# RLEC score: correct=(1-p)^b, incorrect=-p(1-p)^(b-1), abstain=0.
REWARD_B = 1.0
P_EPS = 1e-6
INCLUDE_P_ZERO_IN_SCORE = True
JUDGE_CONCURRENCY = 64

LLM_JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL", "")
LLM_JUDGE_API_KEY = os.environ.get("LLM_JUDGE_API_KEY", "EMPTY")
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "")
LLM_JUDGE_MAX_TOKENS = 128
LLM_JUDGE_ENABLE_THINKING = False
LLM_JUDGE_RETRY = 2

IDK_TEXT = "I am not sure."
# ==================================================================


_TEXT_BRACE_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_llm_client = None


def _extract_boxed(text):
    if not text:
        return None
    results, offset = [], 0
    while True:
        marker = text.find("\\boxed", offset)
        if marker == -1:
            break
        cursor = marker + len("\\boxed")
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            offset = marker + 1
            continue
        depth, start = 0, cursor
        while cursor < len(text):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start + 1 : cursor])
                    break
            cursor += 1
        offset = cursor + 1
    return results[-1] if results else None


def _normalize(answer):
    if answer is None:
        return ""
    if isinstance(answer, (int, float)):
        return str(answer)
    value = str(answer).strip()
    value = _TEXT_BRACE_RE.sub(r"\1", value).replace("\\text", "")
    value = value.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    value = value.replace("$", "").replace("\\%", "").replace("%", "")
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value).strip("{}").rstrip(".")
    if re.fullmatch(r"0*\d+", value):
        value = value.lstrip("0") or "0"
    return value.lower()


def _answers_match(prediction, gold):
    prediction, gold = _normalize(prediction), _normalize(gold)
    if prediction == gold:
        return True
    try:
        return abs(float(prediction) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


def _is_idk(boxed):
    if boxed is None:
        return False
    cleaned = str(boxed)
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _TEXT_BRACE_RE.sub(r"\1", cleaned)
    cleaned = cleaned.replace("\\text", "").lower()
    expected = IDK_TEXT.lower()
    normalized = re.sub(r"\s+", " ", cleaned).strip().strip(".。!! ??,,").strip()
    expected_normalized = (
        re.sub(r"\s+", " ", expected).strip().strip(".。!! ??,,").strip()
    )
    if expected_normalized and expected_normalized in normalized:
        return True
    compact = re.sub(r"\s+", "", cleaned).strip(".。!!??,,")
    expected_compact = re.sub(r"\s+", "", expected).strip(".。!!??,,")
    return bool(expected_compact) and expected_compact in compact


def _judge_string(prediction, gold, question=None):
    del question
    return prediction is not None and _answers_match(prediction, gold)


def _judge_math_verify(prediction, gold, question=None):
    del question
    if prediction is None:
        return False
    try:
        from math_verify import parse as math_parse
        from math_verify import verify as math_verify

        if bool(math_verify(math_parse(f"${gold}$"), math_parse(f"${prediction}$"))):
            return True
    except Exception:
        pass
    return _judge_string(prediction, gold)


def _get_llm_client():
    global _llm_client
    if not LLM_JUDGE_BASE_URL or not LLM_JUDGE_MODEL:
        raise ValueError(
            "llm judge requires LLM_JUDGE_BASE_URL and LLM_JUDGE_MODEL"
        )
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=LLM_JUDGE_BASE_URL,
            api_key=LLM_JUDGE_API_KEY,
        )
    return _llm_client


_LLM_JUDGE_SYSTEM = (
    "You are a grading assistant. Decide whether the predicted answer is "
    "semantically equivalent to the reference gold answer for the given question. "
    "Ignore formatting differences, units notation, LaTeX wrappers, and trivial "
    "rephrasing. Reply with a single token: YES or NO."
)
_LLM_JUDGE_USER_TMPL = (
    "Question:\n{question}\n\nGold answer:\n{gold}\n\n"
    "Predicted answer:\n{prediction}\n\n"
    "Is the predicted answer correct? Reply YES or NO."
)


def _judge_llm(prediction, gold, question=None):
    if prediction is None or not str(prediction).strip():
        return False
    if _judge_string(prediction, gold):
        return True
    prompt = _LLM_JUDGE_USER_TMPL.format(
        question=(question or "")[:2000],
        gold=str(gold)[:1000],
        prediction=str(prediction)[:1000],
    )
    client = _get_llm_client()
    for attempt in range(1, LLM_JUDGE_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=LLM_JUDGE_MAX_TOKENS,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": LLM_JUDGE_ENABLE_THINKING,
                    }
                },
            )
            answer = (response.choices[0].message.content or "").strip().upper()
            if answer.startswith("YES"):
                return True
            if answer.startswith("NO"):
                return False
        except Exception as exc:
            print(
                f"[LLM judge API error] attempt={attempt}/{LLM_JUDGE_RETRY}: "
                f"{exc!r}",
                flush=True,
            )
            if attempt < LLM_JUDGE_RETRY:
                time.sleep(0.5)
    return False


JUDGES = {
    "string": _judge_string,
    "math_verify": _judge_math_verify,
    "llm": _judge_llm,
}


def _grade(record):
    boxed = _extract_boxed(record["output"])
    forced = record["conf"] == 0
    if not forced and _is_idk(boxed):
        return "C"
    if boxed is None:
        return "B"
    judge = JUDGES.get(record["judge"])
    if judge is None:
        raise ValueError(f"unknown judge: {record['judge']!r}")
    return "A" if judge(boxed, record["gold"], record["question"]) else "B"


def _discover_raw_files(inputs):
    files = {}
    for raw_input in inputs:
        path = os.path.abspath(os.path.expanduser(str(raw_input)))
        if os.path.isfile(path):
            candidates = [path]
        elif os.path.isdir(path):
            candidates = []
            for current_dir, dir_names, file_names in os.walk(path):
                dir_names.sort()
                if "raw.jsonl" in file_names:
                    candidates.append(os.path.join(current_dir, "raw.jsonl"))
        else:
            raise FileNotFoundError(f"raw input does not exist: {path}")
        for candidate in candidates:
            files[os.path.realpath(candidate)] = candidate
    paths = sorted(files.values())
    if not paths:
        raise ValueError("no raw.jsonl files found")
    return paths


def _parse_conf(value, where):
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid conf at {where}: {value!r}") from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 100:
        raise ValueError(f"conf must be in [0, 100] at {where}: {confidence}")
    return int(confidence) if confidence.is_integer() else confidence


def _load_records(paths):
    records = []
    seen = set()
    dataset_sources = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                where = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {where}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"record is not an object at {where}")
                required = (
                    "conf",
                    "dataset",
                    "q_idx",
                    "sample",
                    "question",
                    "gold",
                    "judge",
                    "output",
                )
                missing = [field for field in required if field not in record]
                if missing:
                    raise ValueError(f"missing {missing} at {where}")
                record["conf"] = _parse_conf(record["conf"], where)
                try:
                    record["q_idx"] = int(record["q_idx"])
                    record["sample"] = int(record["sample"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid q_idx/sample at {where}") from exc
                if record["q_idx"] < 0 or record["sample"] < 0:
                    raise ValueError(f"negative q_idx/sample at {where}")
                record["dataset"] = str(record["dataset"])
                record["judge"] = str(record["judge"]).strip().lower()
                key = (
                    record["dataset"],
                    record["conf"],
                    record["q_idx"],
                    record["sample"],
                )
                if key in seen:
                    raise ValueError(f"duplicate record key {key} at {where}")
                seen.add(key)
                previous_source = dataset_sources.setdefault(record["dataset"], path)
                if previous_source != path:
                    raise ValueError(
                        f"dataset {record['dataset']!r} appears in multiple raw files: "
                        f"{previous_source}, {path}"
                    )
                records.append(record)
    if not records:
        raise ValueError("raw inputs contain no records")

    models = {
        str(record.get("model", ""))
        for record in records
        if str(record.get("model", ""))
    }
    if len(models) > 1:
        raise ValueError(f"raw inputs contain multiple models: {sorted(models)}")

    identities = {}
    coverage = defaultdict(lambda: defaultdict(set))
    for record in records:
        identity_key = (record["dataset"], record["q_idx"])
        identity = (record["question"], str(record["gold"]), record["judge"])
        previous = identities.setdefault(identity_key, identity)
        if previous != identity:
            raise ValueError(
                f"inconsistent question/gold/judge for {identity_key}: "
                f"{previous!r} != {identity!r}"
            )
        coverage[record["dataset"]][record["conf"]].add(
            (record["q_idx"], record["sample"])
        )
    for dataset, by_conf in coverage.items():
        reference_conf = next(iter(by_conf))
        reference = by_conf[reference_conf]
        for conf, keys in by_conf.items():
            if keys != reference:
                raise ValueError(
                    f"dataset {dataset!r} has inconsistent coverage: "
                    f"conf={reference_conf} has {len(reference)} records, "
                    f"conf={conf} has {len(keys)}"
                )
    return records


def _grade_records(records):
    grades = [None] * len(records)

    def worker(index):
        return index, _grade(records[index])

    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as executor:
        futures = [executor.submit(worker, index) for index in range(len(records))]
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="grade",
            unit="sample",
            dynamic_ncols=True,
        )
        for future in progress:
            index, grade = future.result()
            grades[index] = grade
    return grades


def _build_bins(records, grades):
    counts = defaultdict(lambda: [0, 0, 0])
    grade_index = {"A": 0, "B": 1, "C": 2}
    for record, grade in zip(records, grades):
        counts[(record["dataset"], record["conf"])][grade_index[grade]] += 1

    bins_by_dataset = defaultdict(list)
    for (dataset, conf), (correct, incorrect, abstain) in counts.items():
        total = correct + incorrect + abstain
        attempted = correct + incorrect
        bins_by_dataset[dataset].append(
            {
                "conf": conf,
                "p": conf / 100.0,
                "correct": correct,
                "incorrect": incorrect,
                "abstain": abstain,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "accuracy_attempted": correct / attempted if attempted else 0.0,
                "abstention_rate": abstain / total if total else 0.0,
            }
        )
    for dataset, bins in bins_by_dataset.items():
        bins.sort(key=lambda item: item["p"])
        totals = {item["total"] for item in bins}
        if len(totals) != 1:
            raise ValueError(
                f"dataset {dataset!r} has inconsistent records per confidence bin: "
                f"{sorted(totals)}"
            )
    return dict(sorted(bins_by_dataset.items()))


def _accuracy_at_zero(bins):
    point = next((item for item in bins if item["p"] == 0), None)
    if point is None:
        return None
    return {
        "value": point["accuracy"],
        "correct": point["correct"],
        "total": point["total"],
        "conf": 0,
    }


def _aoc(bins):
    accuracy_key = (
        "accuracy_attempted" if AOC_ACC_MODE == "attempted" else "accuracy"
    )
    points = [
        {
            "p": item["p"],
            "accuracy": item[accuracy_key],
            "overconfidence": max(0.0, item["p"] - item[accuracy_key]),
        }
        for item in bins
    ]
    return {
        "value": sum(item["overconfidence"] for item in points) / len(points),
        "acc_mode": AOC_ACC_MODE,
        "n_bins": len(points),
        "points": points,
    }


def _rewards(p):
    one_minus_p = 1.0 - p
    if REWARD_B == 0.0:
        return 1.0, -p / max(one_minus_p, P_EPS)
    correct_reward = one_minus_p**REWARD_B
    if REWARD_B == 1.0:
        incorrect_reward = -p
    else:
        incorrect_reward = -p * max(one_minus_p, P_EPS) ** (REWARD_B - 1.0)
    return correct_reward, incorrect_reward


def _score(bins):
    points = []
    for item in bins:
        if not INCLUDE_P_ZERO_IN_SCORE and item["p"] == 0:
            continue
        correct_reward, incorrect_reward = _rewards(item["p"])
        value = (
            item["correct"] * correct_reward
            + item["incorrect"] * incorrect_reward
        ) / item["total"]
        points.append(
            {
                "p": item["p"],
                "value": value,
                "correct_reward": correct_reward,
                "incorrect_reward": incorrect_reward,
            }
        )
    if not points:
        raise ValueError("no confidence bins remain for score calculation")
    return {
        "value": sum(item["value"] for item in points) / len(points),
        "reward_b": REWARD_B,
        "include_p_zero": INCLUDE_P_ZERO_IN_SCORE,
        "n_bins": len(points),
        "points": points,
    }


def _pool_bins(bins_by_dataset):
    conf_sets = [
        {item["conf"] for item in bins}
        for bins in bins_by_dataset.values()
    ]
    if not conf_sets or any(conf_set != conf_sets[0] for conf_set in conf_sets[1:]):
        return None
    indexed = {
        dataset: {item["conf"]: item for item in bins}
        for dataset, bins in bins_by_dataset.items()
    }
    pooled = []
    for conf in sorted(conf_sets[0], key=float):
        points = [by_conf[conf] for by_conf in indexed.values()]
        correct = sum(item["correct"] for item in points)
        incorrect = sum(item["incorrect"] for item in points)
        abstain = sum(item["abstain"] for item in points)
        total = correct + incorrect + abstain
        attempted = correct + incorrect
        pooled.append(
            {
                "conf": conf,
                "p": conf / 100.0,
                "correct": correct,
                "incorrect": incorrect,
                "abstain": abstain,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "accuracy_attempted": correct / attempted if attempted else 0.0,
                "abstention_rate": abstain / total if total else 0.0,
            }
        )
    return pooled


def _macro_metric(dataset_reports, name):
    values = [report[name] for report in dataset_reports.values()]
    if any(value is None for value in values):
        return None
    return {
        "value": sum(value["value"] for value in values) / len(values),
        "n_datasets": len(values),
    }


def calculate_metrics(records, grades, source_paths):
    bins_by_dataset = _build_bins(records, grades)
    dataset_reports = {}
    for dataset, bins in bins_by_dataset.items():
        dataset_reports[dataset] = {
            "accuracy": _accuracy_at_zero(bins),
            "aoc": _aoc(bins),
            "score": _score(bins),
            "bins": bins,
        }

    pooled_bins = _pool_bins(bins_by_dataset)
    micro = None
    if pooled_bins is not None:
        micro = {
            "accuracy": _accuracy_at_zero(pooled_bins),
            "aoc": _aoc(pooled_bins),
            "score": _score(pooled_bins),
            "bins": pooled_bins,
        }
    macro = {
        "accuracy": _macro_metric(dataset_reports, "accuracy"),
        "aoc": _macro_metric(dataset_reports, "aoc"),
        "score": _macro_metric(dataset_reports, "score"),
    }
    models = sorted(
        {
            str(record.get("model", ""))
            for record in records
            if str(record.get("model", ""))
        }
    )
    return {
        "metrics": {
            "accuracy": "overall accuracy at p=0",
            "aoc": (
                "mean_p max(0, p-accuracy(p)); accuracy mode is configurable"
            ),
            "score": (
                "mean_p E[(1-p)^b if correct, "
                "-p(1-p)^(b-1) if incorrect, 0 if abstaining]"
            ),
        },
        "config": {
            "aoc_acc_mode": AOC_ACC_MODE,
            "reward_b": REWARD_B,
            "p_eps": P_EPS,
            "include_p_zero_in_score": INCLUDE_P_ZERO_IN_SCORE,
        },
        "models": models,
        "sources": source_paths,
        "datasets": dataset_reports,
        "aggregate": {
            "micro": micro,
            "macro": macro,
            "micro_definition": "pool sample counts per confidence bin",
            "macro_definition": "equal mean over dataset-level metrics",
        },
    }


def _write_outputs(report, output_path):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    fields = (
        "dataset",
        "conf",
        "correct",
        "incorrect",
        "abstain",
        "total",
        "accuracy",
        "accuracy_attempted",
        "abstention_rate",
    )
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for dataset, dataset_report in report["datasets"].items():
            for point in dataset_report["bins"]:
                writer.writerow(
                    {"dataset": dataset, **{field: point[field] for field in fields[1:]}}
                )
    return output_path, csv_path


def _format_metric(value):
    return "-" if value is None else f"{value['value']:.4f}"


def _print_report(report):
    print("dataset\taccuracy\tAOC\tscore")
    for dataset, result in report["datasets"].items():
        print(
            f"{dataset}\t{_format_metric(result['accuracy'])}\t"
            f"{_format_metric(result['aoc'])}\t{_format_metric(result['score'])}"
        )
    micro = report["aggregate"]["micro"]
    macro = report["aggregate"]["macro"]
    if micro is not None:
        print(
            f"micro\t{_format_metric(micro['accuracy'])}\t"
            f"{_format_metric(micro['aoc'])}\t{_format_metric(micro['score'])}"
        )
    print(
        f"macro\t{_format_metric(macro['accuracy'])}\t"
        f"{_format_metric(macro['aoc'])}\t{_format_metric(macro['score'])}"
    )


def _parse_args():
    global OUTPUT_PATH, AOC_ACC_MODE, REWARD_B
    global INCLUDE_P_ZERO_IN_SCORE, JUDGE_CONCURRENCY

    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", default=list(RAW_INPUTS))
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--aoc-acc-mode",
        choices=("attempted", "overall"),
        default=AOC_ACC_MODE,
    )
    parser.add_argument("--reward-b", type=float, default=REWARD_B)
    parser.add_argument("--exclude-p-zero-score", action="store_true")
    parser.add_argument(
        "--judge-concurrency", type=int, default=JUDGE_CONCURRENCY
    )
    args = parser.parse_args()
    if not args.inputs:
        parser.error("provide at least one raw file or directory")
    if not args.output.strip():
        parser.error("--output is required when OUTPUT_PATH is empty")
    if not math.isfinite(args.reward_b) or args.reward_b < 0:
        parser.error("--reward-b must be a finite non-negative number")
    if args.judge_concurrency <= 0:
        parser.error("--judge-concurrency must be positive")
    OUTPUT_PATH = args.output.strip()
    AOC_ACC_MODE = args.aoc_acc_mode
    REWARD_B = args.reward_b
    INCLUDE_P_ZERO_IN_SCORE = not args.exclude_p_zero_score
    JUDGE_CONCURRENCY = args.judge_concurrency
    return args.inputs


def main():
    inputs = _parse_args()
    raw_paths = _discover_raw_files(inputs)
    records = _load_records(raw_paths)
    grades = _grade_records(records)
    report = calculate_metrics(records, grades, raw_paths)
    json_path, csv_path = _write_outputs(report, OUTPUT_PATH)
    _print_report(report)
    print(f"[saved] JSON: {json_path}")
    print(f"[saved] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
