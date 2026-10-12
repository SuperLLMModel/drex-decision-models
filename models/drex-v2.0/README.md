# Drex v2.0

A decision model built on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) with a [Kev](https://github.com/jaredpalmer/kev) pointer head. It takes a `state` and named typed questions ([request format](../../docs/request-format.md)) and returns a probability for every option; nothing is generated. It scores 61.25 on the public Decision Index 0.3.1.

[Weights](https://huggingface.co/nace-ai/drex-v2.0) · [Drex v1.5](../drex-v1.5)

It runs with Python. The default limit is 16,384 tokens per request and it can be raised to 131,072 ([Context length](../../docs/context-length.md)). The bf16 weights take about 70 GB.

<details>
<summary>How it works</summary>

The backbone is a mixture-of-experts `Qwen3_5MoeForConditionalGeneration` with 40 layers, 256 experts and 8 active per token: 35B parameters in total, about 3B active per token. Only its language model is used; the vision tower is dropped on load. Each question is scored in one forward pass over the state plus that question, and the pointer head in `head.pt` turns the result into probabilities. Nothing is generated, so the sampling values in `generation_config.json` do not apply.

</details>

## Performance

### Decision Index 0.3.1 (public only)

| Model | Public index | Raw | Knowledge & Reasoning | Language | Retrieval | Tools | Arts |
|---|---|---|---|---|---|---|---|
| **Drex v2.0** (35B) | **61.25** | 70.55 | 47.9 | 65.3 | 63.0 | 76.5 | 53.9 |
| Drex v1.5 (9B) | 58.08 | 68.08 | 44.6 | 60.4 | 61.7 | 75.0 | 48.8 |
| Jev 1.13.0 | 57.96 | 68.55 | 53.9 | 59.2 | 55.4 | 75.1 | 39.1 |
| Bespoke Nimble 9B v3 | 57.19 | 68.38 | 41.0 | 60.7 | 55.5 | 84.2 | 44.0 |
| Cloudflare clef-flash | 56.15 | 66.53 | 53.0 | 47.2 | 52.2 | 81.9 | 48.4 |
| ezjev 4B s2 | 50.82 | 63.16 | 31.0 | 60.5 | 56.3 | 69.9 | 31.1 |

Public index only: the 37 public benchmarks in five areas, without the private tests. Area columns are chance-corrected skill × 100. The Drex rows are from our own runs of the official kit on the full public suite; the other rows are from the [leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index).

<details>
<summary>JevBench (231 public items)</summary>

| Model | Accuracy | Easy | Standard | Hard |
|---|---|---|---|---|
| Jev 1.13.0 | **86.6%** (200/231) | 100.0% | **98.6%** | **73.0%** |
| **Drex v2.0** | 85.7% (198/231) | 100.0% | 95.8% | **73.0%** |

Choice = top option, yes/no = noul ≥ 0.5, ordinal = top level.

</details>

<details>
<summary>DecideBench v1.1 (400 contrastive decisions)</summary>

Published decision models only (chat LLMs omitted). Few-shot uses the benchmark's examples-in-instructions protocol. Not an official leaderboard submission.

| Entry | Accuracy | Pair accuracy |
|---|---|---|
| Jev (AI Space) | **98.0%** | **96.0%** |
| **Drex v2.0** (few-shot) | 96.5% | 93.5% |
| Drex v2.0 (zero-shot) | 96.3% | 92.5% |
| imajev-4b | 95.0% | 90.5% |
| Clef 27B | 94.8% | 89.5% |
| yev0-4b | 94.3% | 88.5% |
| Bespoke-Nimble-9B | 94.0% | 88.0% |
| TEV | 92.8% | 86.0% |
| Drex v1.5 | 90.0% | 80.5% |
| JevK5 v0.3 | 88.8% | 78.5% |

</details>

<details>
<summary>Email classification</summary>

Held-out addresses. Score is AUC; caught @ 5% is recall at a 5% false-positive rate.

|  | Drex v2.0 | Jev 1.13.0 | Drex v1.5 |
|---|---|---|---|
| AUC | **0.984** | 0.981 | 0.972 |
| Caught @ 5% FP | **89.4%** | 86.4% | 80.3% |

</details>

<details>
<summary>Game arena against Jev</summary>

Eight OpenSpiel games, 32 matches each (16 openings × both colours). Score = (wins + ½ draws) ÷ matches.

|  | W – D – L | Score |
|---|---|---|
| **Drex v2.0 vs Jev** | **132 – 48 – 76** | **60.9%** |
| Drex v1.5 vs Jev | 122 – 47 – 87 | 56.8% |

| Game | Drex v2.0 vs Jev | Drex v1.5 vs Jev |
|---|---|---|
| Chess | 13 – 19 – 0 (70.3%) | 9 – 21 – 2 (60.9%) |
| Four in a Row | 11 – 0 – 21 (34.4%) | 11 – 0 – 21 (34.4%) |
| Reversi | 17 – 0 – 15 (53.1%) | 15 – 1 – 16 (48.4%) |
| Checkers | 20 – 5 – 7 (70.3%) | 21 – 3 – 8 (70.3%) |
| Clobber | 20 – 0 – 12 (62.5%) | 17 – 0 – 15 (53.1%) |
| Lines of Action | 21 – 11 – 0 (82.8%) | 26 – 3 – 3 (85.9%) |
| Dots and Boxes | 15 – 0 – 17 (46.9%) | 18 – 0 – 14 (56.3%) |
| Nine Men's Morris | 15 – 13 – 4 (67.2%) | 5 – 19 – 8 (45.3%) |

</details>

## Python

The server answers `POST /v1/systemone`.

```bash
hf download nace-ai/drex-v2.0 --local-dir weights/drex-v2.0
python serve.py --model weights/drex-v2.0 --port 8000
```

To accept requests longer than 16,384 tokens, set `KEV_CONTEXT` (up to 131,072); see [Context length](../../docs/context-length.md).

## License

Nace.AI Open RAIL-M license, see [LICENSE](LICENSE).
