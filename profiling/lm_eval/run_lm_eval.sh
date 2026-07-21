export BASE_URL="http://localhost:30000/v1/completions,tokenized_requests=False,num_concurrent=1"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B-Base}"
export LM_EVAL_TASK="wikitext"
lm_eval --model local-completions \
    --tasks ${LM_EVAL_TASK} \
    --model_args model=${MODEL_NAME},base_url=${BASE_URL} \
    --batch_size auto

