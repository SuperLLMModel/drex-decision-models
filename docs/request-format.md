# Request format

`POST /v1/systemone` takes a `state` and named `questions`.

| Field | What to put there |
|---|---|
| `state` | The document: a string, object, or list. Nested objects render as indented text. |
| `questions` | A dictionary of named questions. The names come back as the keys of `answers`. |

```json
{
  "state": {
    "ticket": "I was charged twice for the same order. Please refund the extra payment."
  },
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, charges, and refunds",
        "technical": "Bugs and outages",
        "other": "Anything else"
      }
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the customer explicitly ask for a refund?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Routine", "Soon", "Urgent"]
    }
  }
}
```

| Type | You supply | You get back |
|---|---|---|
| `choice` | Named options with short descriptions. Key order is option order. Up to 255 options. | `choice`, a probability per option, and `confidence` |
| `noul` | A yes/no question, with optional `criteria.true` / `criteria.false`. | `noul`, the probability of "yes" (0 to 1) |
| `score` | An ordered scale, lowest first. | A probability-weighted `score`, plus `legend`, per-level `probabilities` and `confidence` |

For `choice`, `confidence` rescales the winning probability above a uniform baseline; for `score`, it measures concentration around the modal level. The local `score` formula approximates the hosted API's.

Response from a bf16 CUDA run of `examples/request.json` with Drex v1.5:

```json
{
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9889,
      "probabilities": { "billing": 0.9926, "technical": 0.0024, "other": 0.005 }
    },
    "refund": { "type": "noul", "noul": 0.9931 },
    "urgency": {
      "type": "score",
      "score": 1.3362,
      "legend": ["Routine", "Soon", "Urgent"],
      "probabilities": { "0": 0.2489, "1": 0.166, "2": 0.5851 },
      "confidence": 0.6681
    }
  },
  "usage": { "input_tokens": 87, "output_tokens": 166 },
  "latency_ms": 860.6
}
```

`usage.input_tokens` is the encoded document plus questions. `usage.output_tokens` counts the serialized answers, not generated text. `latency_ms` is the scoring time.

Requests follow the System One format used by Jev and by [Kev](https://github.com/jaredpalmer/kev), so SDKs and servers written for that format work unchanged.
