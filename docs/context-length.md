# Context length

The models support up to 32,768 tokens. Every runner defaults to 16,384, and the full window is opt-in.

| Runner | Setting |
|---|---|
| Python | `KEV_CONTEXT=32768 python serve.py --model weights/<model> --port 8000` |
| llama-server | `SYSTEMONE_CONTEXT=32768` and a matching `-c`, `-b` and `-ub` (the model page gives the values) |
| Ollama | `PARAMETER num_ctx 32768` in the Modelfile, then `ollama create` again and start the daemon with `SYSTEMONE_CONTEXT=32768` |

`KEV_SERVE_MAX_STATE`, `KEV_SERVE_MAX_BRANCH` and `KEV_SERVE_MAX_PACKED` (Python) and the matching `SYSTEMONE_MAX_STATE`, `SYSTEMONE_MAX_BRANCH` and `SYSTEMONE_MAX_PACKED` (llama-server) override individual caps. None are needed to use the full window.
