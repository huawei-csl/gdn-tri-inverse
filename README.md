# gdn-tri-inverse

Profiling scripts and docker images for Gated DeltaNet-related [pto-kernels](https://github.com/huawei-csl/pto-kernels).

The aim of this repo is to quantify the performance improvements tha can be obtained by using advanced triangular inverse kernels compared to optimized baselines.

## Running with docker (recommended)
The recommended way to use this repo is through the provided Dockerfiles. For more information on the available options and settings see the corresponding scripts.

- Step 1: Build the Docker image (if needed):
```bash
export CANN_VERSION="9.0.0" # Supported: 8.5.0 (with tilelang-ascend) and 9.0.0 (w/o  tilelang-ascend)
bash docker/sgl/build_docker.sh
```

- Step 2: Start the container, test, and profile:
```bash
bash docker/sgl/start_docker_910B2.sh
```
- Step 3 (Optional): Inside the container, you can run the unit tests:
```bash
cd gdn-tri-inv
export NPU_DEVICE="npu:1" # defaults to "npu:0"
python -m pytest -v tests/test_linalg.py
```

- Step 4: Run profiling scripts
```
make profile_tri_inv
```

- Step 5: The results of the profiling can be visualized using the notebooks under `profiling/nbs`.
