FROM jzhou316/ubuntu_1804_py_38_cuda_102_cudnn_8_dev

# pytorch and torchvision
RUN conda install -y -c pytorch pytorch=1.8.1 torchvision=0.9.1 && \
    conda clean -ya

# fairseq: commit at 6/22/2021
# install git -> no need here, git is already there
# RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/pytorch/fairseq.git@7ca8bc12c09d91187d95117094f6b31b3342cd17
RUN pip install tensorboard

# For faster training install NVIDIA's apex library
WORKDIR /opt/nvidia-apex
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" ./

# dependency from local repo setup.py for sm-dataflow
RUN pip install pandas==1.0.0 spacy==2.2.1
RUN python -m spacy download en_core_web_md-2.2.0 --direct

WORKDIR /workspace
RUN chmod -R a+w .

# to run on Azure ML
RUN pip install pip==9.0.0 && \
    pip install ruamel.yaml==0.16 --disable-pip-version-check && \
    pip install --upgrade pip
