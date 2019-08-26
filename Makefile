#
# Simple makefile for object-detection project
#

# Get variables
include Makefile.variables

## Project
COMPONENT_NAME ?= object-detection
export PROJ_ROOT := $(CURDIR)
SUBDIRS =
DOCKERDIR = $(PROJ_ROOT)/docker

## Machine
CORES ?= $(shell nproc)
MAKEFLAGS+="-j $(CORES)"
$(info MAKEFLAGS= $(MAKEFLAGS))

## Python
PYTHON_VERSION ?= 3


.PHONY: all lint flake docker-image docker-image-ncsdk docker-bash docker-bash-ncsdk docker-jupyter docker-stop venv clean clean-all

help:
	@echo
	@echo '  all                   - build and create tsparser main executable.'
	@echo '  lint                  - run clang formating for c++ and flake8 for python'
	@echo '  flake                 - run flake8 on python files.'
	@echo '  docker-image          - builds new docker image with name:tag in Makefile.variables'
	@echo '  docker-image-ncsdk    - builds new docker image for Movidius with NCSDK: ncsdk:v1'
	@echo '  docker-bash           - starts a docker bash interactive container with image in Makefile.variables'
	@echo '  docker-bash-ncsdk     - starts a docker bash interactive container with Movidius NCSDK image.'
	@echo '  docker-jupyter        - starts a docker bash with jupyter notebooks from tensorflow GPU image.'
	@echo '  docker-stop           - stops docker jupyter notebooks container.'
	@echo '  venv                  - build python virtual environment for tensorflow CPU.'
	@echo '  clean                 - deletes build content.'
	@echo '  clean-all             - deletes build content + downloaded 3rd-party.'
	@echo

all: folders $(BUILDDIR)/tsparser

folders: $(BUILDDIR) $(BUILDDIR)/mpeg2vid $(BUILDDIR)/h264

lint: flake

flake:
	flake8

### docker stuff

# Build docker image
docker-image:
	docker build \
		--file=$(DOCKERDIR)/Dockerfile \
		--tag=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VER) \
		--tag=$(DOCKER_IMAGE_NAME):latest docker/


docker-image-ncsdk:
	docker build \
		--file=$(DOCKERDIR)/Dockerfile.NCSDK \
		--tag=$(DOCKER_IMAGE_NCSDK):$(DOCKER_IMAGE_VER_NCSDK) \
		--tag=$(DOCKER_IMAGE_NCSDK):latest docker/


# start tty session inside docker container
docker-bash:
	docker run \
		--runtime=nvidia \
		--rm \
		--interactive \
		--tty=true \
		--publish 8888:8888 \
		--publish 0.0.0.0:6006:6006 \
		--env LOCAL_USER_ID=`id -u ${USER}` \
		--env MODELS_ROOT=/tmp/models \
		--env DISPLAY=unix$$DISPLAY \
		--env TERM=xterm-256color \
		--privileged \
		--volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
		--volume=$$(pwd):/tmp \
		--volume /dev/video0:/dev/video0 \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VER) /bin/bash

docker-bash-ncsdk:
	docker run \
		--rm \
		--interactive \
		--tty=true \
		--env LOCAL_USER_ID=`id -u ${USER}` \
		--env MODELS_ROOT=/home/docker/models \
		--env DISPLAY=unix$$DISPLAY \
		--env TERM=xterm-256color \
		--env QT_X11_NO_MITSHM=1 \
		--privileged \
		--volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
		--volume=$$(pwd):/home/docker/workspace \
		--volume /dev:/dev \
		--net=host \
		$(DOCKER_IMAGE_NCSDK):$(DOCKER_IMAGE_VER_NCSDK) /bin/bash


docker-jupyter:
	docker run --runtime=nvidia \
		--rm \
		--interactive \
		--publish 8888:8888 \
		--volume=$$(pwd):/notebooks/workdir \
		--entrypoint /notebooks/workdir/entrypoint.sh \
		--name tensorflow-jupyter-notebooks \
		tensorflow/tensorflow:nightly-gpu

jupyter: venv
	./venv/bin/jupyter notebook

docker-stop:
	docker stop  tensorflow-jupyter-notebooks

venv:
	virtualenv -p python$(PYTHON_VERSION) $@
	./venv/bin/pip install -r requirements_cpu.txt

clean:
	rm -rf env/
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

### Will force clean download cache & build directory
clean-all: clean
