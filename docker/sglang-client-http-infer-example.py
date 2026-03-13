"""
This is an example of how to use the SGLANG server for inference using HTTP API.
Make sure to start the SGLANG server using the `sglang-server-http-launch.sh` script before running this client script.

You should see the word 'Paris' in the output logs."""

import requests

if __name__ == "__main__":
    model_name = "Qwen/Qwen3.5-0.8B-Base"
    port = 30000
    url = f"http://localhost:{port}/v1/chat/completions"

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    response = requests.post(url, json=data)
    print(response.json())
