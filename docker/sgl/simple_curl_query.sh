export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"
export MODEL_QUERY="which city is the capital of China?"
curl http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
        -d '{  
            "model": "'${MODEL_NAME}'", 
            "messages": [
                {"role": "user", "content": "'"${MODEL_QUERY}"'"}
            ]
        }' | jq '.'
