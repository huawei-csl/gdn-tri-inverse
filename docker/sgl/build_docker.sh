#!/bin/bash

export GDN_TRI_INVERSE_LOCAL_PATH="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/"
export GDN_TRI_INVERSE_COMMIT="${GDN_TRI_INVERSE_COMMIT:-"$(git rev-parse --verify HEAD)"}"

export CANN_VERSION="${CANN_VERSION:-9.0.0}" # Supported: 8.5.0 (with tilelang-ascend) and 9.0.0 (w/o tilelang-ascend)
export DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-gdn-tri-inverse:${CANN_VERSION}-25062026}"
export DOCKERFILE="cann-${CANN_VERSION}.Dockerfile"
# Optional variables (see each Dockerfile):
#SGL_DOCKER_HOSTNAME
#SGL_KERNEL_NPU_BRANCH_OR_TAG
#SGL_KERNEL_NPU_HTTPS_GIT_URL
#TILELANG_ASCEND_COMMIT

pushd $GDN_TRI_INVERSE_LOCAL_PATH \
&& docker build --build-arg GDN_TRI_INVERSE_COMMIT="${GDN_TRI_INVERSE_COMMIT}" \
    -t "${DOCKER_IMAGE_TAG}" \
    -f "${DOCKERFILE}" . \
&& popd
