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
