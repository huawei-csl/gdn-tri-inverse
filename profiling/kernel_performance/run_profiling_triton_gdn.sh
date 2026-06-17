export CMAKE_GENERATOR="Unix Makefiles"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHON_SCRIPT="$(dirname "$(realpath "$0")")"/profile_triton_gdn_npu.py
DATA_PATH="$(dirname "$(realpath "$0")")/../data/Qwen3-Next"
python ${PYTHON_SCRIPT} --chunk-size 64 --input ${DATA_PATH} --inverse-type column-sweep
python ${PYTHON_SCRIPT} --chunk-size 64 --input ${DATA_PATH} --inverse-type cube-rec-unroll
python ${PYTHON_SCRIPT} --chunk-size 64 --input ${DATA_PATH} --inverse-type bsnd-rec-unroll
python ${PYTHON_SCRIPT} --chunk-size 64 --input ${DATA_PATH} --inverse-type triton