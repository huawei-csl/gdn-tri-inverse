DOCKER_IMAGE_TAG="gdn-main-cann-8.5.0"

drun() {

docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume ~/sgl-kernel-npu/:/tmp/sgl-kernel-npu/ --volume ~/sglang/:/tmp/sglang/ \
    --name sglang-${USER} \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/ "$@"
}

drun --env "HF_ENDPOINT=https://hf-mirror.com" ${DOCKER_IMAGE_TAG} /usr/bin/bash
