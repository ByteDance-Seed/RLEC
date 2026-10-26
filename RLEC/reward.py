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

"""RLEC reward with a linear stability-coefficient schedule."""

import os
import random
import re
import threading
from decimal import Decimal, InvalidOperation


IDK_TEXT = os.environ.get("IDK_TEXT", "I am not sure.").strip().lower()
PRINT_SAMPLE_RATE = float(os.environ.get("PRINT_SAMPLE_RATE", "0.001"))
WARMUP_STEPS = float(os.environ.get("WARMUP_STEPS", "585"))
CALLS_PER_STEP = float(os.environ.get("CALLS_PER_STEP", "256"))
P_EPS = float(os.environ.get("P_EPS", "1e-6"))

_print_lock = threading.Lock()
_call_lock = threading.Lock()
_call_count = 0


def _is_escaped(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _extract_boxed(text):
    """Extract the final ``\\boxed{...}``, including nested braces."""
    if not text:
        return None
    start = text.rfind(r"\boxed")
    if start < 0:
        return None
    open_brace = text.find("{", start + len(r"\boxed"))
    if open_brace < 0:
        return None

    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{" and not _is_escaped(text, index):
            depth += 1
        elif char == "}" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index].strip()
    return None


_LATEX_TEXT_RE = re.compile(
    r"\\(?:text|textrm|mathrm|operatorname)\s*\{([^{}]*)\}", re.IGNORECASE
)
_NUMBER_RE = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)


def _unwrap_latex_text(value):
    previous = None
    while value != previous:
        previous = value
        value = _LATEX_TEXT_RE.sub(r"\1", value)
    return value


def _normalize(answer):
    if answer is None:
        return None
    value = _unwrap_latex_text(str(answer).strip())
    value = value.replace("\\$", "$").strip()
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(".。").rstrip().casefold()


def _parse_integer(answer):
    if answer is None:
        return None
    value = _unwrap_latex_text(str(answer).strip())
    value = value.replace("\\$", "").replace("$", "")
    value = value.replace("\\!", "").replace("\\,", "").replace(",", "")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".。") if not value.endswith(".0") else value
    if not value or _NUMBER_RE.fullmatch(value) is None:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return int(number)


def _answers_match(prediction, target):
    normalized_prediction = _normalize(prediction)
    normalized_target = _normalize(target)
    if normalized_prediction is None or normalized_target is None:
        return False
    if normalized_prediction == normalized_target:
        return True

    target_integer = _parse_integer(target)
    prediction_integer = _parse_integer(prediction)
    return (
        target_integer is not None
        and prediction_integer is not None
        and prediction_integer == target_integer
    )


def _is_idk(boxed):
    if boxed is None:
        return False
    normalized = _normalize(boxed)
    expected = _normalize(IDK_TEXT)
    return bool(expected) and expected in (normalized or "")


def _advance_and_get_step(count=1):
    global _call_count
    with _call_lock:
        start = _call_count
        _call_count += count
    if CALLS_PER_STEP <= 0:
        return None
    return int(start // CALLS_PER_STEP)


def _stability_coefficient(step):
    """Increase b linearly from zero to one over training."""
    if step is None or WARMUP_STEPS <= 1:
        return 1.0
    return min(max(float(step), 0.0) / (WARMUP_STEPS - 1), 1.0)


def _rlec_rewards(confidence, coefficient):
    """Return rewards for correct and incorrect answers.

    Correct: ``(1-p)^b``; incorrect: ``-p(1-p)^(b-1)``. Abstention
    receives zero. For every b this preserves the answer rule q(x) > p.
    """
    try:
        p = float(confidence) if confidence is not None else 0.5
    except (TypeError, ValueError):
        p = 0.5
    p = min(1.0, max(0.0, p))
    one_minus_p = 1.0 - p
    correct = one_minus_p**coefficient
    incorrect = -p if coefficient == 1 else -p * max(one_minus_p, P_EPS) ** (coefficient - 1)
    return correct, incorrect


def _is_train(data_source):
    return "__" not in str(data_source or "")


def _score_one(data_source, solution, target, extra_info, coefficient):
    kind = str((extra_info or {}).get("kind", "") or "").lower()
    if kind not in {"qa", "conf"}:
        kind = "conf" if (extra_info or {}).get("confidence") is not None else "qa"

    boxed = _extract_boxed(solution)
    correct = boxed is not None and _answers_match(boxed, target)
    if kind == "qa":
        grade = "A" if correct else "B"
        score = 1.0 if correct else 0.0
    elif boxed is not None and _is_idk(boxed):
        grade, score = "C", 0.0
    else:
        correct_reward, incorrect_reward = _rlec_rewards(
            (extra_info or {}).get("confidence"), coefficient
        )
        grade = "A" if correct else "B"
        score = correct_reward if correct else incorrect_reward
    return grade, boxed, score, kind


def _result(grade, score):
    return {
        "score": score,
        "correct": float(grade == "A"),
        "incorrect": float(grade == "B"),
        "not_attempted": float(grade == "C"),
    }


def _maybe_print(data_source, solution, target, extra_info, grade, prediction, score, step, coefficient):
    if PRINT_SAMPLE_RATE <= 0 or random.random() >= PRINT_SAMPLE_RATE:
        return
    with _print_lock:
        print(
            "\n[RLEC reward sample] "
            f"source={data_source} grade={grade} score={score:+g} "
            f"p={(extra_info or {}).get('confidence')} step={step} b={coefficient:.4f}\n"
            f"  question: {(extra_info or {}).get('question', '')}\n"
            f"  target:   {target}\n"
            f"  output:   {solution}\n"
            f"  parsed:   {prediction}",
            flush=True,
        )


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    step = _advance_and_get_step() if _is_train(data_source) else None
    coefficient = _stability_coefficient(step)
    grade, prediction, score, _ = _score_one(
        data_source, solution_str, str(ground_truth), extra_info, coefficient
    )
    _maybe_print(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        grade,
        prediction,
        score,
        step,
        coefficient,
    )
    return _result(grade, score)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
    count = len(solution_strs)
    extra_infos = extra_infos if extra_infos is not None else [None] * count
    train_count = sum(_is_train(source) for source in data_sources)
    step = _advance_and_get_step(train_count) if train_count else None
    train_coefficient = _stability_coefficient(step)

    results = []
    for index in range(count):
        extra_info = extra_infos[index] or {}
        coefficient = train_coefficient if _is_train(data_sources[index]) else 1.0
        grade, prediction, score, _ = _score_one(
            data_sources[index],
            solution_strs[index],
            str(ground_truths[index]),
            extra_info,
            coefficient,
        )
        _maybe_print(
            data_sources[index],
            solution_strs[index],
            ground_truths[index],
            extra_info,
            grade,
            prediction,
            score,
            step if _is_train(data_sources[index]) else None,
            coefficient,
        )
        results.append(_result(grade, score))
    return results
