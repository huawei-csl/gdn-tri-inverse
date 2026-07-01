# Docker

This directory contains Docker files and scripts.

## SGL
The `sgl/` subdirectory contains all the scripts based which use `sglang` images as a base image.
There are two Dockerfiles currently, summarized in the following table:

Base Image | CANN 9.0.0  | sgl-kernel-npu | torch / torch-npu | tilelang-ascend | triton-ascend | pto-kernels |

|                 | CANN 8.5.0                   | CANN 9.0.0                          |
| :-------------- | ---------------------------: | ----------------------------------: |
| Base Image      | sglang:v0.5.9-cann8.5.0-910b | sglang:v0.5.13.post1-cann9.0.0-910b |
| sgl-kernel-npu  | 2026.3.1 **                  | 2026.3.2                            |
| torch           | 2.8.0+cpu                    | 2.10.0+cpu                          |
| torch-npu       | 2.8.0.post2                  | 2.10.0                              |
| tilelang-ascend | commit `d4736eb`             | --                                  |
| triton-ascend   | 3.2.0                        | 3.2.1                               |
| pto-kernels     | 0.1.5                        | 0.1.5                               |

** Obtained from `https://github.com/gioelegott/sgl-kernel-npu.git@6-triinv-integrate-tri_inv_cube_col_sweep-kernel`