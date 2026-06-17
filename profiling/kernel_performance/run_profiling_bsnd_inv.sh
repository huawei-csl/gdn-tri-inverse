source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TRITON_ALL_BLOCKS_PARALLEL=1
PYTHON_SCRIPT="$(dirname "$(realpath "$0")")"/profile_bsnd_inv_npu.py
echo ${PYTHON_SCRIPT}
python ${PYTHON_SCRIPT} --chunk-size 16
python ${PYTHON_SCRIPT} --chunk-size 32
python ${PYTHON_SCRIPT} --chunk-size 64
python ${PYTHON_SCRIPT} --chunk-size 128
