### Run lm_eval benchmark on a running sglang server

- Step 1: Install in a local environment the requirements, e.g. `pip install -r requirements.txt`
- Step 2 (Optional): Launch an http server inside a docker container, e.g. by executing the script `docker/sgl/start_sglang_http_server.sh` (see the script for mode details).
- Step 3: Run `profiling/lm_eval/run_lm_eval.sh`. Optional settings: `BASE_URL`, `MODEL_NAME`, `LM_EVAL_TASK` (defaults to `wikitext`)