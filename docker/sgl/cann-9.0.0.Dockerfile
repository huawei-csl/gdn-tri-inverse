ARG SGL_DOCKER_HOSTNAME=""

# Downloader image
FROM ${SGL_DOCKER_HOSTNAME}quay.io/ascend/sglang:v0.5.13.post1-cann9.0.0-910b AS downloader
ARG TILELANG_ASCEND_COMMIT="63bef06" # v0.1.1.010-release
ARG SGL_KERNEL_NPU_BRANCH_OR_TAG="2026.6.2"
ARG SGL_KERNEL_NPU_HTTPS_GIT_URL="https://github.com/sgl-project/sgl-kernel-npu.git"

ENV TILELANG_ASCEND_COMMIT=${TILELANG_ASCEND_COMMIT}
ENV SGL_KERNEL_NPU_HTTPS_GIT_URL=${SGL_KERNEL_NPU_HTTPS_GIT_URL}
ENV SGL_KERNEL_NPU_BRANCH_OR_TAG=${SGL_KERNEL_NPU_BRANCH_OR_TAG}

RUN git clone ${SGL_KERNEL_NPU_HTTPS_GIT_URL} \
    && cd sgl-kernel-npu \
    && git checkout ${SGL_KERNEL_NPU_BRANCH_OR_TAG} \
    && bash build.sh -a kernels \
    && cp output/sgl_kernel_npu*.whl /tmp/ \
    && cd ../ \
    && rm -rf sgl-kernel-npu

# RUN git clone --recursive https://github.com/tile-ai/tilelang-ascend.git \
#     && cd tilelang-ascend \
#     && git checkout ${TILELANG_ASCEND_COMMIT} \
#     && ./build_wheel_ascend.sh \
#     && cp dist/tilelang-*.whl /tmp/

# Main image
FROM ${SGL_DOCKER_HOSTNAME}quay.io/ascend/sglang:v0.5.13.post1-cann9.0.0-910b AS main
ARG GDN_TRI_INVERSE_COMMIT

ENV GDN_TRI_INVERSE_COMMIT=${GDN_TRI_INVERSE_COMMIT}

# Install build dependencies
RUN pip install pyyaml setuptools pytest

# Install sgl-kernel-npu and tilelang-ascend
COPY --from=downloader /tmp/*.whl /workspace/
RUN pip install /workspace/*.whl

# Install gdn-tri-inverse
RUN pip install pyyaml setuptools pytest
# RUN pip install torch-npu==2.10.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install pto-kernels==0.1.4 
RUN cd /workspace/ \
    && git clone https://github.com/huawei-csl/gdn-tri-inverse.git \
    && cd gdn-tri-inverse \
    && git checkout ${GDN_TRI_INVERSE_COMMIT} \
    && pip install -v --no-deps . 

# Set up environment for runtime

ADD ./set_env.sh /etc/profile.d/02-env-ascend-and-tilelang.sh
RUN cat /etc/profile.d/02-env-ascend-and-tilelang.sh >> /root/.bashrc

CMD ["/bin/bash"]
