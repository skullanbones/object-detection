##################################################
#
# Docker image for tensorflow object detection API
#
##################################################

FROM nvidia/cuda:9.0-base-ubuntu16.04

### install packages ###
RUN apt-get update && apt-get -y --no-install-recommends install \
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
    less


## Python base 2.7
RUN apt-get update && apt-get install --yes --no-install-recommends
RUN apt-get install --yes --no-install-recommends python-pip
RUN apt-get install --yes --no-install-recommends virtualenv
RUN apt-get install --yes --no-install-recommends python-setuptools
RUN apt-get install --yes --no-install-recommends python-dev

## Install tensorflow dependencies
RUN apt-get install --yes --no-install-recommends cuda-command-line-tools-9-0
RUN apt-get install --yes --no-install-recommends cuda-cublas-9-0
RUN apt-get install --yes --no-install-recommends cuda-cusolver-9-0
RUN apt-get install --yes --no-install-recommends libcudnn7=7.2.1.38-1+cuda9.0
RUN apt-get install --yes --no-install-recommends cuda-cufft-9-0
RUN apt-get install --yes --no-install-recommends cuda-curand-9-0

## Python packages 
# TODO use requirements.txt
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install utils
RUN pip install Pillow lxml six
RUN pip install tensorflow-gpu
RUN pip install matplotlib

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

# Entrypoint
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start up jupyter notebooks
#CMD ["/run_jupyter.sh", "--allow-root"]

# Tensorboard
EXPOSE 6006/tcp
