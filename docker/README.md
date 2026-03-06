# Docker

This directory contains Docker configuration files for the GDN Triangular Inverse project.

## Contents

- `Dockerfile` - Container image definition that contains sglang, tilelang-ascend, triton and pto-kernels.
- `sglang-eval.md` - Instructions on how to run sglang HTTP server and client.
- `start_docker_9102/4.sh` - scripts to start the docker container
- `sglang-server-http-launch.sh` - script to start slglang server (to be run inside docker)
- `sglang-server-http-infer-example.sh` - script to query/prompt sglang server over HTTP.
- `build_docker.sh` - builds the docker image
