ARG SGL_DOCKER_HOSTNAME=""

# Downloader image
FROM ${SGL_DOCKER_HOSTNAME}lmsysorg/sglang:main-cann8.5.0-910b AS downloader
ARG SGL_KERNEL_NPU_BRANCH="6-triinv-integrate-tri_inv_cube_col_sweep-kernel"
ARG SGL_KERNEL_NPU_HTTPS_GIT_URL="https://github.com/gioelegott/sgl-kernel-npu.git"

ENV SGL_KERNEL_NPU_HTTPS_GIT_URL=${SGL_KERNEL_NPU_HTTPS_GIT_URL}
ENV SGL_KERNEL_NPU_BRANCH=${SGL_KERNEL_NPU_BRANCH}

RUN git clone ${SGL_KERNEL_NPU_HTTPS_GIT_URL} --branch ${SGL_KERNEL_NPU_BRANCH} \
    && cd sgl-kernel-npu \
    && bash build.sh -a kernels \
    && cp output/sgl_kernel_npu*.whl /tmp/ \
    && cd ../ \
    && rm -rf sgl-kernel-npu

# Main image
FROM ${SGL_DOCKER_HOSTNAME}lmsysorg/sglang:main-cann8.5.0-910b AS main
ARG TILELANG_ASCEND_COMMIT="d4736eb"
ENV TILELANG_ASCEND_COMMIT=${TILELANG_ASCEND_COMMIT}
ARG GDN_TRI_INVERSE_LOCAL_PATH="."

# Install build dependencies
RUN pip install pyyaml setuptools pytest
RUN pip uninstall triton -y

# Install tilelang
RUN git clone --recursive https://github.com/tile-ai/tilelang-ascend.git \
    && cd tilelang-ascend \
    && git reset --hard ${TILELANG_ASCEND_COMMIT} \
    && bash install_ascend.sh

# Install sgl-kernel-npu and pto-kernels
COPY --from=downloader /tmp/*.whl /workspace/
RUN pip install --force-reinstall /workspace/*.whl

# Install gdn-tri-inverse
ADD ${GDN_TRI_INVERSE_LOCAL_PATH} /workspace/gdn-tri-inverse
RUN cd /workspace/gdn-tri-inverse \
    && export CMAKE_GENERATOR="Unix Makefiles" \
    && . /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && pip install --force-reinstall -v . --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://test.pypi.org/simple/

# Set up environment for runtime
COPY ${GDN_TRI_INVERSE_LOCAL_PATH}/docker/set_env.sh /etc/profile.d/02-env-ascend-and-tilelang.sh
RUN cat /etc/profile.d/02-env-ascend-and-tilelang.sh >> /root/.bashrc

CMD ["/bin/bash"]
