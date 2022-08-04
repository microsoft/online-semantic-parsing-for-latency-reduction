### Build Base Docker Images

[Here](./ubuntu_cpu_gpu/docker_build.sh): the image is ready to build PyTorch from source.

PyTorch is not yet downloaded nor installed.

A Python with specific version is installed. Conda is available.

## Source

https://github.com/pytorch/pytorch/blob/53bc6f79f320e3ce080fcd14727d18389619a229/docker/pytorch/ubuntu_cpu_gpu/Dockerfile


### Build PyTorch + fairseq Docker Images

[Here](./pytorch_fairseq/docker_build.sh): the image has PyTorch installation, plus fairseq.

Also has other custom installations, including tensorboard and Nvidia apex (for fairseq), etc.

## Reference

https://github.com/pytorch/pytorch/blob/v1.8.1/docker/pytorch/Dockerfile

https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.8.1-cuda11.1-ubuntu20.04/Dockerfile
