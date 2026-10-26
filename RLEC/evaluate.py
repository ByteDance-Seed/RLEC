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

"""Generate and save raw explicit-confidence evaluation responses.

This file only calls the evaluated model and writes ``raw.jsonl``. It does not
grade answers or calculate metrics; use ``metrics.py`` for that second stage.
Configure the blank paths below or supply their command-line equivalents.
"""

import argparse
import json
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


# ========================== Configuration ==========================
BASE_URL = os.environ.get("EVAL_BASE_URL", "")
API_KEY = os.environ.get("EVAL_API_KEY", "EMPTY")
MODEL = os.environ.get("EVAL_MODEL", "")
RUN_NAME = os.environ.get("EVAL_RUN_NAME", "")
RESULTS_ROOT = os.environ.get("EVAL_RESULTS_ROOT", "")

# name -> (JSON path, judge used later by metrics.py, samples per question)
DATASET_FILES = {
    # "dataset_name": ("", "string", 1),
}

CONF_LIST = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
DEFAULT_N_SAMPLES = 16
N_SAMPLES_OVERRIDE = None
MAX_TOKENS = 4096
TEMPERATURE = 1.0
TOP_P = 0.95
CONCURRENCY = 2000
REQUEST_RETRY = 3
TASK_SHUFFLE_SEED = 20260811

IDK_TEXT = "I am not sure."
THINK_INSTRUCTION = (
    "Work through this problem step by step, then honestly assess how "
    "confident you are that your solution is correct."
)
CONF_INSTRUCTION_TMPL = (
    "Only give a final answer if you are at least {p_pct} confident that it "
    "is correct; otherwise do not guess and answer with \\boxed{{"
    + IDK_TEXT
    + "}} instead."
)
BOX_INSTRUCTION = (
    "Put your final answer inside \\boxed{}. "
    "If you are not confident enough, put \\boxed{" + IDK_TEXT + "} instead."
)
PLAIN_BOX_INSTRUCTION = (
    "Work through this problem step by step. Put your final answer inside \\boxed{}."
)
# ==================================================================


_UNSAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._-]+")
_client = None


def _safe_component(value, label):
    original = str(value).strip()
    safe = _UNSAFE_COMPONENT_RE.sub("_", original).strip("._")
    if not safe:
        raise ValueError(f"{label} cannot be mapped to a valid path component")
    return safe


def parse_dataset_configs(dataset_files, default_n_samples=1, n_override=None):
    parsed = {}
    for dataset_name, value in dataset_files.items():
        if isinstance(value, (list, tuple)):
            if len(value) == 3:
                path, judge_name, n_samples = value
            elif len(value) == 2:
                path, judge_name = value
                n_samples = default_n_samples
            else:
                raise ValueError(
                    f"DATASET_FILES[{dataset_name!r}] must be "
                    "(path, judge, n_samples)"
                )
        else:
            path, judge_name, n_samples = value, "string", default_n_samples

        if n_override is not None:
            n_samples = n_override
        try:
            n_samples = int(n_samples)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid n_samples for dataset {dataset_name!r}: {n_samples!r}"
            ) from exc
        path = str(path).strip()
        if not path:
            raise ValueError(f"dataset path is empty: {dataset_name!r}")
        if n_samples <= 0:
            raise ValueError(
                f"n_samples must be positive for {dataset_name!r}: {n_samples}"
            )
        parsed[dataset_name] = {
            "path": path,
            "judge": str(judge_name).strip().lower(),
            "n_samples": n_samples,
        }
    return parsed


def build_output_paths(dataset_names):
    if not RESULTS_ROOT:
        raise ValueError("RESULTS_ROOT is empty")
    safe_run_name = _safe_component(RUN_NAME, "RUN_NAME")
    output_paths = {}
    used_components = {}
    for dataset_name in dataset_names:
        safe_dataset = _safe_component(dataset_name, "dataset name")
        previous = used_components.get(safe_dataset)
        if previous is not None and previous != dataset_name:
            raise ValueError(
                f"dataset names {previous!r} and {dataset_name!r} map to "
                f"the same directory {safe_dataset!r}"
            )
        used_components[safe_dataset] = dataset_name
        output_dir = os.path.normpath(
            os.path.join(RESULTS_ROOT, safe_dataset, safe_run_name)
        )
        output_paths[dataset_name] = {
            "output_dir": output_dir,
            "raw_path": os.path.join(output_dir, "raw.jsonl"),
        }
    return output_paths


def read_json_qa(path):
    with open(path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    rows = []
    for record in data:
        question = record.get("question") if isinstance(record, dict) else None
        if question is None and isinstance(record, dict):
            question = record.get("problem")
        gold = record.get("answer") if isinstance(record, dict) else None
        if question is None or gold is None:
            continue
        question = str(question).strip()
        if question:
            rows.append({"question": question, "answer": str(gold)})
    if not rows:
        raise ValueError(f"dataset has no usable question/answer records: {path}")
    return rows


def _conf_instruction(confidence):
    confidence = min(0.99, max(0.01, confidence))
    return CONF_INSTRUCTION_TMPL.format(p_pct=f"{round(confidence * 100)}%")


def build_prompt_text(question, conf):
    question = (question or "").strip()
    if conf == 0:
        return question + "\n\n" + PLAIN_BOX_INSTRUCTION
    return (
        question
        + "\n\n"
        + THINK_INSTRUCTION
        + " "
        + _conf_instruction(conf / 100.0)
        + " "
        + BOX_INSTRUCTION
    )


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def call_once(prompt_text):
    """Return ``(reasoning + content, error)`` after bounded retries."""
    client = _get_client()
    last_error = None
    for attempt in range(1, REQUEST_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning", None) or ""
            content = message.content or ""
            return (str(reasoning) + "\n" + str(content)).strip(), None
        except Exception as exc:  # API implementations expose different errors.
            last_error = repr(exc)
            print(
                f"[model API error] attempt={attempt}/{REQUEST_RETRY}: {last_error}",
                flush=True,
            )
            if attempt < REQUEST_RETRY:
                time.sleep(1.0)
    return f"__ERROR__ {last_error}", last_error


def _temporary_path(path, label):
    directory, filename = os.path.split(path)
    token = uuid.uuid4().hex[:12]
    return os.path.join(directory, f".{filename}.{label}.{token}.tmp")


def _read_raw_record(line, path, line_number):
    where = f"{path}:{line_number}"
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {where}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"raw record is not an object at {where}")
    try:
        record["conf"] = int(record["conf"])
        record["q_idx"] = int(record["q_idx"])
        record["sample"] = int(record["sample"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid conf/q_idx/sample at {where}") from exc
    return record


def _merge_single_conf(
    old_path,
    new_path,
    merged_path,
    dataset_name,
    target_conf,
    dataset_size,
    n_samples,
):
    """Atomically prepare one-bin replacement while preserving other bins."""
    seen = set()
    new_keys = set()
    expected_new_keys = {
        (question_index, sample_index)
        for question_index in range(dataset_size)
        for sample_index in range(n_samples)
    }

    def consume(path, output, is_new):
        with open(path, "r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                record = _read_raw_record(line, path, line_number)
                if str(record.get("dataset")) != dataset_name:
                    raise ValueError(
                        f"dataset mismatch at {path}:{line_number}: "
                        f"{record.get('dataset')!r}"
                    )
                conf = record["conf"]
                if is_new and conf != target_conf:
                    raise ValueError(
                        f"new raw contains conf={conf}, expected {target_conf}"
                    )
                if not is_new and conf == target_conf:
                    continue
                key = (conf, record["q_idx"], record["sample"])
                if key in seen:
                    raise ValueError(f"duplicate raw record key: {key}")
                seen.add(key)
                if is_new:
                    new_keys.add((record["q_idx"], record["sample"]))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(merged_path, "w", encoding="utf-8") as output:
        consume(old_path, output, is_new=False)
        consume(new_path, output, is_new=True)
    if new_keys != expected_new_keys:
        missing = len(expected_new_keys - new_keys)
        extra = len(new_keys - expected_new_keys)
        raise ValueError(
            f"replacement bin is incomplete: missing={missing}, extra={extra}"
        )


def _validate_configuration(dataset_configs):
    if not BASE_URL:
        raise ValueError("BASE_URL is empty; set EVAL_BASE_URL or edit the file")
    if not MODEL:
        raise ValueError("MODEL is empty; set EVAL_MODEL or edit the file")
    if not RUN_NAME:
        raise ValueError("RUN_NAME is empty; set EVAL_RUN_NAME or edit the file")
    if not RESULTS_ROOT:
        raise ValueError(
            "RESULTS_ROOT is empty; set EVAL_RESULTS_ROOT or edit the file"
        )
    if not dataset_configs:
        raise ValueError("DATASET_FILES is empty; configure at least one dataset")
    if not CONF_LIST:
        raise ValueError("CONF_LIST cannot be empty")
    if len(CONF_LIST) != len(set(CONF_LIST)):
        raise ValueError(f"CONF_LIST contains duplicates: {CONF_LIST}")
    if any(conf < 0 or conf > 100 for conf in CONF_LIST):
        raise ValueError(f"confidence values must be in [0, 100]: {CONF_LIST}")


def run():
    dataset_configs = parse_dataset_configs(
        DATASET_FILES,
        default_n_samples=DEFAULT_N_SAMPLES,
        n_override=N_SAMPLES_OVERRIDE,
    )
    _validate_configuration(dataset_configs)

    datasets = {}
    for name, config in dataset_configs.items():
        records = read_json_qa(config["path"])
        datasets[name] = records
        print(
            f"[dataset] {name}: {len(records)} questions x "
            f"{config['n_samples']} samples; judge={config['judge']}"
        )

    output_paths = build_output_paths(datasets)
    for paths in output_paths.values():
        os.makedirs(paths["output_dir"], exist_ok=True)

    tasks = []
    for conf in CONF_LIST:
        for dataset_name, records in datasets.items():
            n_samples = dataset_configs[dataset_name]["n_samples"]
            for question_index, record in enumerate(records):
                prompt = build_prompt_text(record["question"], conf)
                for sample_index in range(n_samples):
                    tasks.append(
                        (
                            conf,
                            dataset_name,
                            question_index,
                            sample_index,
                            record["question"],
                            record["answer"],
                            prompt,
                        )
                    )
    random.Random(TASK_SHUFFLE_SEED).shuffle(tasks)
    print(f"[generation] {len(tasks)} requests, concurrency={CONCURRENCY}")

    temporary_paths = {
        name: _temporary_path(paths["raw_path"], "generation")
        for name, paths in output_paths.items()
    }
    files = {
        name: open(path, "w", encoding="utf-8")
        for name, path in temporary_paths.items()
    }

    def worker(task):
        conf, dataset_name, q_idx, sample, question, gold, prompt = task
        output, error = call_once(prompt)
        return {
            "conf": conf,
            "dataset": dataset_name,
            "q_idx": q_idx,
            "sample": sample,
            "question": question,
            "gold": gold,
            "judge": dataset_configs[dataset_name]["judge"],
            "model": MODEL,
            "output": output,
            "error": error,
        }

    completed = False
    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [executor.submit(worker, task) for task in tasks]
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="generate",
                unit="req",
                dynamic_ncols=True,
            )
            for done, future in enumerate(progress, 1):
                record = future.result()
                files[record["dataset"]].write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                progress.set_postfix_str(f"saved={done}")
        completed = True
    finally:
        for output_file in files.values():
            output_file.close()
        if not completed:
            for path in temporary_paths.values():
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    single_conf = len(CONF_LIST) == 1
    target_conf = CONF_LIST[0] if single_conf else None
    for dataset_name, paths in output_paths.items():
        raw_path = paths["raw_path"]
        new_path = temporary_paths[dataset_name]
        if single_conf and os.path.isfile(raw_path) and os.path.getsize(raw_path) > 0:
            merged_path = _temporary_path(raw_path, f"merge-conf-{target_conf}")
            try:
                _merge_single_conf(
                    old_path=raw_path,
                    new_path=new_path,
                    merged_path=merged_path,
                    dataset_name=dataset_name,
                    target_conf=target_conf,
                    dataset_size=len(datasets[dataset_name]),
                    n_samples=dataset_configs[dataset_name]["n_samples"],
                )
                os.replace(merged_path, raw_path)
            finally:
                for path in (new_path, merged_path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
        else:
            os.replace(new_path, raw_path)
        print(f"[saved] {dataset_name}: {raw_path}")


def _parse_args():
    global BASE_URL, MODEL, RUN_NAME, RESULTS_ROOT
    global N_SAMPLES_OVERRIDE, CONCURRENCY, CONF_LIST

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument(
        "--conf",
        default="",
        help="comma-separated confidence percentages; one value replaces that bin",
    )
    args = parser.parse_args()
    BASE_URL = args.base_url.strip()
    MODEL = args.model.strip()
    RUN_NAME = args.run_name.strip()
    RESULTS_ROOT = args.results_root.strip()
    N_SAMPLES_OVERRIDE = args.n
    CONCURRENCY = args.concurrency
    if args.conf.strip():
        CONF_LIST = sorted(
            int(value.strip())
            for value in args.conf.split(",")
            if value.strip()
        )
    if CONCURRENCY <= 0:
        parser.error("--concurrency must be positive")


if __name__ == "__main__":
    _parse_args()
    run()
