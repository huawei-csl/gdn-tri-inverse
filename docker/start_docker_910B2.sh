#!/bin/bash
#
# start_docker_910B2.sh
#
# Description:
#   Start a Docker container with an SGLANG friendly image for triangular inverse benchmarking on Ascend 910B2 server.
#   The script mounts the local cache of the user into the container so that the model weights are not downloaded every time the container is started.
#
# Usage:
#   ./start_docker_910B2.sh
#

# On 910B2 server, we MUST build this image by yourself.
DOCKER_IMAGE_TAG="registry.gitlab.huaweirc.ch/zrc-von-neumann-lab/tcuscan/gdn-tri-inverse:8.5.0-b47b974"

drun() {

docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume $(pwd):/workspace/gdn-tri-inv-repo \
    --name sglang-${USER} \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/ "$@"
}

drun --env "HF_ENDPOINT=https://hf-mirror.com" ${DOCKER_IMAGE_TAG} /usr/bin/bash
