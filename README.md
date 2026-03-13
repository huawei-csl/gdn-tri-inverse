# gdn-tri-inverse
Code to perform end-to-end for Gated Delta Nets using different triangular inversion algorithms.

## TLDR
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CMAKE_GENERATOR="Unix Makefiles" && pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu
pytest tests/
```


## Running sglang on Docker

Steps to run a client-server example of sglang on Docker.

```bash
./docker/start_docker_910B4.sh # or 910B2
```

Inside the docker container, type
```bash
cd gdn-tri-inv-repo/
./docker/sglang-server-http-launch.sh
```

Wait to make sure the server is running. Type `ctrl-z` and `bg` to send process in bg

```bash
python ./docker/sglang-client-http-infer-example.py
```

You should see

```
{'id': 'f051d47f3f5045a2a7aa28edb6b9d971', 'object': 'chat.completion', 'created': 1773388679, 'model': 'Qwen/Qwen3.5-0.8B-Base', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The capital of France is **Paris**. Founded in 751 AD by Charlemagne, it is a major cultural and political hub in Europe and the world.\n', 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 248044}], 'usage': {'prompt_tokens': 19, 'total_tokens': 56, 'completion_tokens': 37, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}
```
