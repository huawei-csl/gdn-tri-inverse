source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TRITON_ALL_BLOCKS_PARALLEL=1
python profiling/profile_bsnd_inv_npu.py --chunk-size 16
python profiling/profile_bsnd_inv_npu.py --chunk-size 32
python profiling/profile_bsnd_inv_npu.py --chunk-size 64
python profiling/profile_bsnd_inv_npu.py --chunk-size 128
