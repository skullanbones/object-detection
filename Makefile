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


OBJS = $(patsubst %.cc,$(BUILDDIR)/%.o,$(SRCS))

$(info OBJS is: $(OBJS))

.PHONY: all clean lint flake docker-image docker-bash test gtests run clang-tidy clang-format unit-test component-tests

help:
	@echo
	@echo '  all                   - build and create tsparser main executable.'
	@echo '  lint                  - run clang formating for c++ and flake8 for python'
	@echo '  flake                 - run flake8 on python files.'
	@echo '  run                   - run tsparser for bbc_one.ts asset and write elementary streams.'
	@echo '  docker-image          - builds new docker image with name:tag in Makefile.'
	@echo '  docker-bash           - starts a docker bash session with settings in makefile.'
	@echo '  docker-run            - starts a docker bash session with settings in makefile.'
	@echo '  docker-jupyter        - starts a docker bash with jupyter notebooks from tensorflow image.'
	@echo '  env                   - build python virtual environment for pytest.'
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
		--tag=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VER) docker/

# start tty session inside docker container
docker-bash:
	docker run \
		--rm \
		--interactive \
		--tty=true \
		--env LOCAL_USER_ID=`id -u ${USER}` \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VER) /bin/bash
#		--volume=$$(pwd):/workdir \		

docker-run:
	docker run \
		--rm \
		--interactive \
		--env LOCAL_USER_ID=`id -u ${USER}` \
		--publish 8888:8888 \
		--volume=$$(pwd):/workdir \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VER)

docker-image-od:
	docker build \
		--file $(DOCKERDIR)/Dockerfile.sofwerx \
		-t sofwerx/od:1.5.0-devel-gpu . --no-cache=true

# Didnt work
#docker-image-tf:
#	docker build \
#		--file=Dockerfile.tf \
#		--tag=heliconwave/tf:v1 .

docker-jupyter:
	docker run --runtime=nvidia \
		--rm \
		--interactive \
		--publish 8888:8888 \
		--volume=$$(pwd):/notebooks/workdir \
		tensorflow/tensorflow:nightly-gpu	

### component tests

# Doesnt work
#docker-nvidia:
#	docker run \
#		--runtime=nvidia \
#		-it \
#		-p 8888:8888 \
#		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VER)

env:
	virtualenv -p python$(PYTHON_VERSION) $@
	./env/bin/pip install -r component_tests/requirements.txt


clean:
	rm -rf env/
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

### Will force clean download cache & build directory
clean-all: clean
