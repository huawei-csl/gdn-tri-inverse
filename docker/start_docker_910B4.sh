#!/bin/bash
#
# start_docker_910B4.sh
#
# Description:
#   Start a Docker container with an SGLANG friendly image for triangular inverse benchmarking on Ascend NPU 910B4 environment containing 2 NPUs.
#   The script mounts the local cache of the user into the container so that the model weights are not downloaded every time the container is started.
#
# Usage:
#   ./start_docker_910B4.sh
#

DOCKER_IMAGE_TAG="registry.gitlab.huaweirc.ch/zrc-von-neumann-lab/tcuscan/gdn-tri-inverse:8.5.0-8644320"

drun() {

docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume $(pwd):/workspace/gdn-tri-inv-repo \
    --name sglang-${USER} \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/ "$@"
}

drun ${DOCKER_IMAGE_TAG} /usr/bin/bash
