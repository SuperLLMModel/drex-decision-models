# Drex decision models

🚀 **Drex v1.5 is out!** Rank 1 on Decision Index 0.2.1 with a score of 58.28, and it runs on llama.cpp and Ollama too. [Model page](models/drex-v1.5) · [Weights](https://huggingface.co/nace-ai/drex-v1.5) · [Performance](models/drex-v1.5#performance)

🎉 **Drex DLM is out!** A diffusion-language-model decision model with a Q8_0 GGUF for Apple silicon and CUDA. [Model page](models/drex-dlm) · [BF16](https://huggingface.co/nace-ai/drex-dlm) · [Q8_0](https://huggingface.co/nace-ai/drex-dlm-Q8_0)

Open decision models from [Nace.AI](https://www.nace.ai/). Pass a `state` and named typed questions (yes/no, multiple choice or rating scale) and get a probability for every option back; nothing is generated. Every model serves the `POST /v1/systemone` API of the hosted [Drex API](https://drex.nace.ai/docs), so you can run it yourself.

[Hosted API docs](https://drex.nace.ai/docs) · [Agent skill](https://github.com/nace-ai/drex-agent-skill)

## Models

| Model | Backbone | Weights | llama.cpp and Ollama branch | License |
|---|---|---|---|---|
| [Drex v1.5](models/drex-v1.5) | [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) (distilled Qwen 3.5 9B) | [nace-ai/drex-v1.5](https://huggingface.co/nace-ai/drex-v1.5) | `drex-v1.5` | [Nace.AI Open RAIL-M](models/drex-v1.5/LICENSE) |
| [Drex DLM](models/drex-dlm) | Efficient-DLM-8B | [nace-ai/drex-dlm](https://huggingface.co/nace-ai/drex-dlm) | `edlm` (llama.cpp), `nace-edlm` (Ollama) | [CC BY-NC 4.0](models/drex-dlm/LICENSE.md) |

## Runners

Every model runs in three ways and answers the same `POST /v1/systemone` request.

| Runner | Hardware | Use it for |
|---|---|---|
| Python (`serve.py`) | NVIDIA GPU | The reference implementation, batched requests |
| [llama.cpp](https://github.com/nace-ai/llama.cpp) fork | CUDA, Metal, CPU | A GGUF served by `llama-server`, including Q8_0 |
| [Ollama](https://github.com/nace-ai/ollama) fork | CUDA, Metal, CPU | `ollama create` and a local daemon |

The commands below use Drex v1.5. Each model page has the exact steps, build presets and tested hardware.

## Python

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

## llama.cpp

Build the fork, convert the weights to GGUF and serve them. Add `-DGGML_CUDA=ON` to the first `cmake` command on NVIDIA, and `-ngl 99` to `llama-server` to use the GPU.

```bash
git clone --branch drex-v1.5 --single-branch https://github.com/nace-ai/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j --target llama-server llama-quantize

python llama.cpp/convert_hf_to_gguf.py weights/drex-v1.5 --no-mtp --outtype bf16 \
  --outfile weights/drex-v1.5/drex-v1.5.gguf
llama.cpp/build/bin/llama-server -m weights/drex-v1.5/drex-v1.5.gguf \
  --embedding --pooling none -np 2 -c 32768 -b 16384 -ub 2048 --port 8000
```

The server takes the same request as `serve.py`. `llama-quantize` makes a Q8_0 GGUF of about 9.5 GB.

## Ollama

The Nace fork of Ollama runs the model as a `decision` capability and starts `llama-server` itself. Build the fork against the llama.cpp checkout above. These are the macOS commands; the [model page](models/drex-v1.5/README.md#ollama) has the CPU and CUDA presets.

```bash
git clone --branch drex-v1.5 --single-branch https://github.com/nace-ai/ollama.git
cd ollama
export OLLAMA_LLAMA_CPP_SOURCE="$PWD/../llama.cpp"
cmake -S llama/server --preset darwin
cmake --build build/llama-server-darwin --target llama-server --parallel 8
GOTOOLCHAIN=auto go build -trimpath -o ollama .
cd ..
```

Import the GGUF, start the daemon and leave it running:

```bash
cat > weights/drex-v1.5/Modelfile <<'EOF'
FROM ./drex-v1.5.gguf
CAPABILITY decision
PARAMETER num_ctx 16384
EOF
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_LLAMA_SERVER="$PWD/ollama/build/llama-server-darwin/bin/llama-server" ollama/ollama serve
```

In another terminal, import the model and send the example. Ollama needs `"model"` in the request, which the `jq` filter adds:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama/ollama create drex-v1.5 -f weights/drex-v1.5/Modelfile
jq '. + {model: "drex-v1.5"}' examples/request.json | curl http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d @-
```

## Docs

- [Request format](docs/request-format.md): fields, question types, responses and batches.
- [Context length](docs/context-length.md): default 16,384 tokens, up to 131,072 for Drex v1.5 and 32,768 for Drex DLM.

## License

Code: Apache-2.0, see [LICENSE](LICENSE). Model weights: see each model's page ([Drex v1.5](https://huggingface.co/nace-ai/drex-v1.5)).
