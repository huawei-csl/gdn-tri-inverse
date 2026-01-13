source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Fixes: "RuntimeError: grid should be less than 65536! You can try "export TRITON_ALL_BLOCKS_PARALLEL=1" to avoid this problem."
export TRITON_ALL_BLOCKS_PARALLEL=1
python profiling/profile_tri_inv_npu.py --chunk-size 16
python profiling/profile_tri_inv_npu.py --chunk-size 32
python profiling/profile_tri_inv_npu.py --chunk-size 64
python profiling/profile_tri_inv_npu.py --chunk-size 128
