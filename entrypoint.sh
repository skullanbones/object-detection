#!/bin/bash

echo "from entrypoint"

unzip /notebook/workdir/protoc-3.2.0-linux-x86_64.zip  -d protoc3
cp /notebooks/workdir/protoc3/bin/* /usr/local/bin/
cp /notebooks/workdir/protoc3/include/* /usr/local/include/

protoc --version

cd /notebooks/workdir/models/research
protoc object_detection/protos/*.proto --python_out=.
echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc

# Startup jupyter
jupyter notebook --allow-root

# Change directory
cd /notebooks/workdir/models/research/object_detection
