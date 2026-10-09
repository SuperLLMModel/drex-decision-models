"""Score a System One request with a Drex model.

    pip install -r requirements.txt
    python inference.py --model weights/drex-v1.5
    python inference.py --model weights/drex-v1.5 --request examples/request.json

The model returns a probability for every option in one forward pass.
A state or question past the context limit returns an error, so the score covers the whole document.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from kev.checkpoint import load
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_PACKED, SERVE_MAX_STATE

EXAMPLE = {
    "state": {
        "ticket": "I was charged twice for the same order. Please refund the extra payment.",
    },
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, charges, and refunds",
                "technical": "Bugs and outages",
                "other": "Anything else",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["Routine", "Soon", "Urgent"],
        },
    },
}


def _encode(tok, model, request):
    if not isinstance(request, dict) or "requests" in request:
        raise ValueError("each decision must be an object with state and questions, without a requests wrapper")
    rec, meta = to_record(SystemOneRequest.model_validate(request))
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
    return enc, meta


def _result(tok, enc, meta, probs, latency_ms):
    if len(probs) != len(meta) or any(len(row) != len(question["keys"]) or
                                      any(not math.isfinite(value) for value in row)
                                      for row, question in zip(probs, meta)):
        raise RuntimeError("model readout is incomplete or non-finite")
    answers = to_answers(probs, meta)
    return {
        "answers": answers,
        "usage": {"input_tokens": len(enc["ids"]), "output_tokens": _run_scorer(output_tokens, tok, answers)},
        "latency_ms": latency_ms,
    }


def _run_scorer(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:
        raise RuntimeError("model scoring failed") from exc


def decide(tok, model, request):
    enc, meta = _encode(tok, model, request)
    started = time.perf_counter()
    with torch.no_grad():
        probs = [p.float().tolist() for p in _run_scorer(model.probs, enc)]
    return _result(tok, enc, meta, probs, round((time.perf_counter() - started) * 1000, 1))


def decide_batch(tok, model, requests):
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")
    if len(requests) > 1024:
        raise ValueError("requests exceeds 1024 items")
    prepared = [_encode(tok, model, request) for request in requests]
    results = [None] * len(prepared)
    index = 0
    while index < len(prepared):
        enc, _meta = prepared[index]
        if len(enc["ids"]) > SERVE_MAX_PACKED:
            started = time.perf_counter()
            with torch.no_grad():
                probs = [p.float().tolist() for p in _run_scorer(model.probs, enc)]
            results[index] = _result(tok, enc, _meta, probs, round((time.perf_counter() - started) * 1000, 1))
            index += 1
            continue
        end = index
        longest = 0
        # Bound padded tokens and the quadratic attention mask before batching.
        while end < len(prepared) and end - index < 8:
            size = len(prepared[end][0]["ids"])
            if size > SERVE_MAX_PACKED:
                break
            padded = (max(longest, size) + 127) // 128 * 128
            count = end - index + 1
            if count * padded > SERVE_MAX_PACKED or count * padded * padded > SERVE_MAX_PACKED ** 2:
                break
            longest = max(longest, size)
            end += 1
        if end == index:
            end = index + 1
        group = prepared[index:end]
        started = time.perf_counter()
        with torch.no_grad():
            logits = _run_scorer(model.forward_batch, [item[0] for item in group])
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if len(logits) != len(group):
            raise RuntimeError("batch readout did not return one result per request")
        for offset, ((enc, meta), row) in enumerate(zip(group, logits)):
            probs = [torch.softmax(item.float(), dim=-1).tolist() for item in row]
            results[index + offset] = _result(tok, enc, meta, probs, latency_ms)
        index = end
    return results


def main():
    parser = argparse.ArgumentParser(description="Score one System One request with a Drex model.")
    parser.add_argument("--model", required=True, help="Local checkpoint directory")
    parser.add_argument("--request", help="JSON file with one System One request")
    args = parser.parse_args()
    tok, model = load(args.model)
    request = json.loads(Path(args.request).read_text()) if args.request else EXAMPLE
    print(json.dumps(decide(tok, model, request), indent=2))


if __name__ == "__main__":
    main()
