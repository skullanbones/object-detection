#!/bin/bash

echo "from entrypoint"

unzip /notebooks/workdir/3rd-party/protoc-3.2.0-linux-x86_64.zip  -d protoc3
cp ./protoc3/bin/* /usr/local/bin/
cp ./protoc3/include/* /usr/local/include/

protoc --version

cd /notebooks/workdir/models/research
protoc object_detection/protos/*.proto --python_out=.
echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc

# Change directory
cd /notebooks/workdir/

# upgrade tensorflow
pip install tensorflow --upgrade

# Startup jupyter
jupyter notebook --allow-root
