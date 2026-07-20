curl http://127.0.0.1:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
        -d '{  
            "model": "Qwen/Qwen3.5-0/.8B-Base", 
            "messages": [
                {"role": "user", "content": "which city is the capital of China? "}
            ]
        }' | jq '.'