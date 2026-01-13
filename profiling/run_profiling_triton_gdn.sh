export CMAKE_GENERATOR="Unix Makefiles"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python profiling/profile_triton_gdn_npu.py --chunk-size 64 --inverse-type column-sweep
python profiling/profile_triton_gdn_npu.py --chunk-size 64 --inverse-type cube-rec-unroll
python profiling/profile_triton_gdn_npu.py --chunk-size 64 --inverse-type bsnd-rec-unroll
python profiling/profile_triton_gdn_npu.py --chunk-size 64 --inverse-type triton