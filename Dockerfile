FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git python3 python3-pip vim cmake 

# Make image smaller by not caching downloaded pip pkgs
ARG PIP_NO_CACHE_DIR=1

# Install pytorch for example, and ensure sim works with all our required pPkgs
ARG TORCH_SCATTER_VERSION=2.0.9
ARG TORCH_VERSION=1.11.0+cu113
ARG TORCH_SPARSE_VERSION=0.6


# Pytorch and torch_geometric w/ deps
RUN python3 -m pip install torch==${TORCH_VERSION} -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN python3 -m pip install torchtext==0.12.0
RUN python3 -m pip install torch-scatter==${TORCH_SCATTER_VERSION} -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
RUN python3 -m pip install torch-sparse==${TORCH_SPARSE_VERSION} -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
RUN python3 -m pip install git+https://github.com/pyg-team/pytorch_geometric.git


RUN pip install seaborn
RUN pip install wandb
RUN pip install shapely
RUN pip install torch summary
RUN pip install easydict

WORKDIR /home

