# Drex decision models

🚀 **Drex v1.5 is out!** Rank 1 on Decision Index 0.2.1 with a score of 58.28, and it runs on llama.cpp and Ollama too. [Model page](models/drex-v1.5) · [Weights](https://huggingface.co/nace-ai/drex-v1.5) · [Performance](models/drex-v1.5#performance)

🎉 **Drex DLM is out!** A diffusion-language-model decision model with a Q8_0 GGUF for Apple silicon and CUDA. [Model page](models/drex-dlm) · [BF16](https://huggingface.co/nace-ai/drex-dlm) · [Q8_0](https://huggingface.co/nace-ai/drex-dlm-Q8_0)

Open decision models from [Nace.AI](https://www.nace.ai/). Pass a `state` and named typed questions (yes/no, multiple choice or rating scale) and get a probability for every option back; nothing is generated. Every model serves the `POST /v1/systemone` API of the hosted [Drex API](https://drex.nace.ai/docs), so you can run it yourself.

[Hosted API docs](https://drex.nace.ai/docs) · [Agent skill](https://github.com/nace-ai/drex-agent-skill)

## Models

| Model | Backbone | Weights | License |
|---|---|---|---|
| [Drex v1.5](models/drex-v1.5) | Qwen3.5-based, 9B | [nace-ai/drex-v1.5](https://huggingface.co/nace-ai/drex-v1.5) | [Nace.AI Open RAIL-M](models/drex-v1.5/LICENSE) |
| [Drex DLM](models/drex-dlm) | Efficient-DLM-8B | [nace-ai/drex-dlm](https://huggingface.co/nace-ai/drex-dlm) | [CC BY-NC 4.0](models/drex-dlm/LICENSE.md) |

Each model page has its benchmarks, hardware and the llama.cpp and Ollama steps.

## Quick start

```bash
git clone https://github.com/nace-ai/drex-decision-models.git
cd drex-decision-models
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download nace-ai/drex-v1.5 --local-dir weights/drex-v1.5
python inference.py --model weights/drex-v1.5 --request examples/request.json
```

To serve it instead:

```bash
python serve.py --model weights/drex-v1.5 --port 8000
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

`GET /health` returns `{"status": "ok", "model": "drex-v1.5"}` once the weights are loaded. Replace `drex-v1.5` with any model above.

## Docs

- [Request format](docs/request-format.md): fields, question types, responses and batches.
- [Context length](docs/context-length.md): the 32,768-token window and its settings.

## License

Code: Apache-2.0, see [LICENSE](LICENSE). Model weights: see each model's page.
