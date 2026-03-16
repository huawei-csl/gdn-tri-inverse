#!/bin/bash

export GDN_TRI_INVERSE_LOCAL_PATH="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/../"
export SGL_DOCKER_HOSTNAME="${SGL_DOCKER_HOSTNAME:-}"
export SGL_KERNEL_NPU_BRANCH="${SGL_KERNEL_NPU_BRANCH:-6-triinv-integrate-tri_inv_cube_col_sweep-kernel}"
export SGL_KERNEL_NPU_HTTPS_GIT_URL="${SGL_KERNEL_NPU_HTTPS_GIT_URL:-https://github.com/gioelegott/sgl-kernel-npu.git}"
export TILELANG_ASCEND_COMMIT="${TILELANG_ASCEND_COMMIT:-d4736eb}"
export DOCKER_IMAGE_TAG="registry.gitlab.huaweirc.ch/zrc-von-neumann-lab/tcuscan/gdn-tri-inverse:8.5.0-$(git rev-parse --short HEAD)"

pushd $GDN_TRI_INVERSE_LOCAL_PATH \
&& docker build --build-arg SGL_DOCKER_HOSTNAME="${SGL_DOCKER_HOSTNAME}" \
    --build-arg SGL_KERNEL_NPU_BRANCH="${SGL_KERNEL_NPU_BRANCH}" \
    --build-arg SGL_KERNEL_NPU_HTTPS_GIT_URL="${SGL_KERNEL_NPU_HTTPS_GIT_URL}" \
    --build-arg TILELANG_ASCEND_COMMIT="${TILELANG_ASCEND_COMMIT}" \
    -t "${DOCKER_IMAGE_TAG}" \
    -f docker/Dockerfile . \
&& popd
