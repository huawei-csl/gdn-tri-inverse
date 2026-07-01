ARG SGL_DOCKER_HOSTNAME=""

# Downloader image
FROM ${SGL_DOCKER_HOSTNAME}quay.io/ascend/sglang:v0.5.9-cann8.5.0-910b AS downloader
ARG SGL_KERNEL_NPU_BRANCH_OR_TAG="6-triinv-integrate-tri_inv_cube_col_sweep-kernel" # "2026.6.2" # for cube-col-sweep: 6-triinv-integrate-tri_inv_cube_col_sweep-kernel
ARG SGL_KERNEL_NPU_HTTPS_GIT_URL="https://github.com/gioelegott/sgl-kernel-npu.git" # "https://github.com/sgl-project/sgl-kernel-npu.git" # for cube-col-sweep: https://github.com/gioelegott/sgl-kernel-npu.git
 
ENV SGL_KERNEL_NPU_HTTPS_GIT_URL=${SGL_KERNEL_NPU_HTTPS_GIT_URL}
ENV SGL_KERNEL_NPU_BRANCH_OR_TAG=${SGL_KERNEL_NPU_BRANCH_OR_TAG}

RUN git clone ${SGL_KERNEL_NPU_HTTPS_GIT_URL} \
    && cd sgl-kernel-npu \
    && git checkout ${SGL_KERNEL_NPU_BRANCH_OR_TAG} \
    && bash build.sh -a kernels \
    && cp output/sgl_kernel_npu*.whl /tmp/ \
    && cd ../ \
    && rm -rf sgl-kernel-npu

# Main image
FROM ${SGL_DOCKER_HOSTNAME}quay.io/ascend/sglang:v0.5.9-cann8.5.0-910b AS main
ARG TILELANG_ASCEND_COMMIT="d4736eb"
ARG GDN_TRI_INVERSE_COMMIT

ENV TILELANG_ASCEND_COMMIT=${TILELANG_ASCEND_COMMIT}
ENV GDN_TRI_INVERSE_COMMIT=${GDN_TRI_INVERSE_COMMIT}

# Install build dependencies
RUN pip install pyyaml setuptools pytest
RUN pip uninstall triton -y

# Install tilelang
RUN git clone --recursive https://github.com/tile-ai/tilelang-ascend.git \
    && cd tilelang-ascend \
    && git reset --hard ${TILELANG_ASCEND_COMMIT} \
    && bash install_ascend.sh

# Install sgl-kernel-npu
COPY --from=downloader /tmp/*.whl /workspace/
RUN pip install --force-reinstall /workspace/*.whl

# Install gdn-tri-inverse
RUN cd /workspace/ \
    && git clone https://github.com/huawei-csl/gdn-tri-inverse.git \
    && cd gdn-tri-inverse \
    && git checkout ${GDN_TRI_INVERSE_COMMIT} \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && export CMAKE_GENERATOR="Unix Makefiles" \
    && pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://test.pypi.org/simple/

# Set up environment for runtime

ADD ./set_env.sh /etc/profile.d/02-env-ascend-and-tilelang.sh
RUN cat /etc/profile.d/02-env-ascend-and-tilelang.sh >> /root/.bashrc

CMD ["/bin/bash"]
