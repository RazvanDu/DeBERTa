#!/bin/bash

# https://devblogs.nvidia.com/gpu-containers-runtime/
[[ -z $(dpkg --list | grep nvidia-docker2) ]] && (
	curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
	curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |sudo tee --allow-unauthenticated /etc/apt/sources.list.d/nvidia-docker.list
	sudo apt-get update
	sudo apt-get install -y nvidia-docker2
  sudo pkill -SIGHUP dockerd
)

docker run --runtime nvidia -it --rm -v `pwd`:/DeBERTa --workdir /DeBERTa/applications/glue/  bagai/deberta bash
