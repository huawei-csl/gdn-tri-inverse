source /usr/local/Ascend/ascend-toolkit/set_env.sh

python profiling/profile_tri_inv_npu.py --chunk-size 16
python profiling/profile_tri_inv_npu.py --chunk-size 32
python profiling/profile_tri_inv_npu.py --chunk-size 64
python profiling/profile_tri_inv_npu.py --chunk-size 128
