# gdn-tri-inverse

Code to perform end-to-end for Gated Delta Nets using different triangular inversion algorithms.

## Running GDN with docker (recommended)

Step 1: Build the Docker image (if needed):
```bash
bash docker/build_docker.sh
```

Step 2: Start the container, test, and profile:
```bash
bash docker/start_docker_910B4.sh
```
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd gdn-tri-inv-repo
make test_tri_inv
make profile_tri_inv
```

Step 3 (optional): Compile again and test
```bash
make install
make test_tri_inv
```

## Running GDN baremetal (only for python version <= 3.11)

Step 1: Install `gdn-tri-inverse`:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
make install
```

Step 2: Install `sgl-kernel-npu`:
```bash
git clone https://github.com/gioelegott/sgl-kernel-npu.git --branch checkout 6-triinv-integrate-tri_inv_cube_col_sweep-kernel
cd sgl-kernel-npu
bash build.sh -a kernels
pip install --force-reinstall output/sgl_kernel_npu*.whl
cd ..
```

Step 3: install `tilelang-ascend`
[WIP]

Step 4: Run the tests:
```bash
make test_tri_inv
```

Step 5: Run profiling:
```bash
make profile_tri_inv
```

## Profiling
The profiling scripts that compare all the methods are inside `profiling/`.
E.g., to compare only the triangular inverse methods run:
```bash
./profiling/run_profiling_tri_inv.sh
```
Optionally, before running the script, the specific device that will be used can be specified:
```bash
export GDN_TRI_INVERSE_NPU_DEVICE="npu:4" # Optional, set NPU device to run profiling on.
```