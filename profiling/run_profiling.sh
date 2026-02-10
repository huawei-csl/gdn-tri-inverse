export CMAKE_GENERATOR="Unix Makefiles"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python profiling/profile_gdn_npu.py --chunk-size 64 --inverse-type baseline
python profiling/profile_gdn_npu.py --chunk-size 64 --inverse-type column-sweep
python profiling/profile_gdn_npu.py --chunk-size 64 --inverse-type cube-column-sweep
python profiling/profile_gdn_npu.py --chunk-size 64 --inverse-type cube-rec-unroll
python profiling/profile_gdn_npu.py --chunk-size 64 --inverse-type triton

python profiling/profile_gdn_npu.py --torch-profiler --chunk-size 64 --inverse-type baseline
python profiling/profile_gdn_npu.py --torch-profiler --chunk-size 64 --inverse-type column-sweep
python profiling/profile_gdn_npu.py --torch-profiler --chunk-size 64 --inverse-type cube-column-sweep
python profiling/profile_gdn_npu.py --torch-profiler --chunk-size 64 --inverse-type cube-rec-unroll
python profiling/profile_gdn_npu.py --torch-profiler --chunk-size 64 --inverse-type triton
