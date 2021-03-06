##################################################
#
# Docker image for tensorflow object detection API
#
##################################################

FROM nvidia/cuda:10.0-base-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

### install packages ###
RUN apt-get update && apt-get -y --no-install-recommends install \
    apt-utils \
    ca-certificates \
    curl \
    gnupg2 \
    gosu \
    python-qt4 \
    git \
    wget \
    nano \
    autotools-dev \
    libtool \
    autoconf \
    automake \
    unzip \
    less \
    sudo \
    iputils-ping \
    sox \
    libsndfile1 \
    flac \
## Python base 3
    python3 \
    python3-pip \
    virtualenv \
    python3-setuptools \
    python3-dev \
## Install tensorflow dependencies
    cuda-command-line-tools-10-0 \
    cuda-cublas-10-0 \
    cuda-cusolver-10-0 \
    libcudnn7=7.6.1.34-1+cuda10.1 \
    cuda-cufft-10-0 \
    cuda-curand-10-0 \
    cuda-cusparse-10-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install protocol buffers tool
RUN curl -L -o protoc-3.6.1-linux-x86_64.zip https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
RUN rm protoc-3.6.1-linux-x86_64.zip
RUN mv ./protoc3/bin/* /usr/local/bin/ 
RUN mv ./protoc3/include/* /usr/local/include/ 
RUN rm -rf ./protoc3
#RUN protoc object_detection/protos/*.proto --python_out=.

# Add root password
RUN echo "root:Docker!" | chpasswd

ENV LANG=en_US.UTF-8
RUN export LC_ALL="en_US.UTF-8"
RUN export LC_CTYPE="en_US.UTF-8"

## Python packages 

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
COPY requirements_gpu.txt /tmp
RUN pip3 install -r /tmp/requirements_gpu.txt

# Entrypoint
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start up jupyter notebooks
#CMD ["/run_jupyter.sh", "--allow-root"]

# Tensorboard
EXPOSE 6006/tcp
