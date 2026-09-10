export BASE_URL="http://localhost:30000/v1/completions,tokenized_requests=False,num_concurrent=1"
export OUTPUT_PATH="${OUTPUT_PATH:-./results}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B-Base}"
export LM_EVAL_TASKS="longbench_single" # comma-separated tasks, e.g. 
# wikitext,mmlu_anatomy,mmlu_abstract_algebra,mmlu_college_computer_science,mmlu_college_mathematics
export NUM_ITERS=1
for i in $(seq 1 $NUM_ITERS)
do
    time lm_eval --model local-completions \
        --tasks ${LM_EVAL_TASKS} \
        --model_args model=${MODEL_NAME},base_url=${BASE_URL} \
        --batch_size auto \
        --output_path ${OUTPUT_PATH} > /dev/null 2>&1
done
