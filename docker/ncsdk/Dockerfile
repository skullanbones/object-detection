######################################################################
#
# Simple Dockerfile for building with Intel Neural Compute Stick SDK.
#
######################################################################

FROM ubuntu:18.04 as base

### Create user account ###

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y --no-install-recommends install \
    ca-certificates \
    curl \
    gnupg2 \
    gosu \
    git \
    wget \
    build-essential \
    sudo \
    less \
    lsb-release \
    udev \
    usbutils \
    libv4l-dev \
    nano \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

### Builder image ###


WORKDIR /home/docker

RUN echo "alias ll='ls -l'" > /home/docker/.bashrc

### Install Intel NCSDK
RUN git clone http://github.com/movidius/ncsdk --depth=1
WORKDIR /home/docker/ncsdk
RUN chmod +x *.sh
RUN make install
RUN chmod +x *.sh
RUN yes | ./install-opencv.sh

# Python packages
RUN pip3 install scikit-image

# Add root password
RUN echo "root:Docker!" | chpasswd

# entrypoint
COPY ./entrypoint_ncsdk.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

ENV LANG=en_US.UTF-8
WORKDIR /home/docker/workspace
