##################################################
#
# Docker image for tensorflow object detection API
#
##################################################

FROM tensorflow/tensorflow:1.12.0-devel-gpu as base

### Create user account ###
FROM base as user
RUN apt-get update && apt-get -y --no-install-recommends install \
    ca-certificates \
    curl

RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# Entrypoint
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

### Builder image ###
FROM user as build

## Packages
RUN apt-get update && apt-get install --yes --no-install-recommends
RUN apt-get install --yes --no-install-recommends python-qt4
RUN apt-get install --yes --no-install-recommends git
RUN apt-get install --yes --no-install-recommends wget
RUN apt-get install --yes --no-install-recommends nano
RUN apt-get install --yes --no-install-recommends autotools-dev
RUN apt-get install --yes --no-install-recommends libtool
RUN apt-get install --yes --no-install-recommends autoconf
RUN apt-get install --yes --no-install-recommends automake 

## Python 
RUN pip install opencv-python
RUN pip install utils
RUN pip install --upgrade protobuf
RUN pip install Pillow lxml

#  && cd /notebooks && rm -rf protobuf/
#COPY ../notebooks /workdir/notebooks

# TODO git clone TF models
# TODO workdir
#WORKDIR /workdir/tensorflow
#RUN git clone https://github.com/sozercan/models
#WORKDIR /workdir/tensorflow/models/research
#ENV PYTHONPATH "$PYTHONPATH:/workdir/tensorflow/models/research:/workdir/tensorflow/models/research/slim"
#ENV PYTHON_HOME "$PYTHON_HOME:/workdir/tensorflow/models/research:/workdir/tensorflow/models/research/slim"

# Install protocol buffers tool
RUN curl -L -o protoc-3.6.1-linux-x86_64.zip https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
RUN unzip protoc-3.6.1-linux-x86_64.zip
RUN rm protoc-3.6.1-linux-x86_64.zip
RUN mv ./bin/* /usr/local/bin/
RUN mv ./include/* /usr/local/include/
#RUN protoc object_detection/protos/*.proto --python_out=.

# Add root password
RUN echo "root:Docker!" | chpasswd

ENV LANG=en_US.UTF-8

## TODO REMOVE
# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

# Start up jupyter notebooks
#CMD ["/run_jupyter.sh", "--allow-root"]
