# Computer vision with AI
This repo contains a docker image with the purpose of process isolation for training and running tensorflow calculations on your NVIDIA GPU hardware isolating NVIDIA CUDA. This will simplify the installation and save considerable amount of time instead of having to install it on your host. It's also easy to reuse this image on AWS to simplify training on new AWS instances. Check requirements for what is required by the host. Please check the [`TRAINING.md`](TRAINING.md) to get started with training.

There is also a docker image available for NCS1 development.

## Host requirements
| Host                   | Version minimal requirement   |
|------------------------|-------------------------------|
| Ubuntu                 | 18.04                         |
| Docker                 | >18.x                         |
| NVIDIA Docker version  | >v2                           |
| Nvidia driver          | > nvidia-driver-430           |


## Tensorflow object detection
Tensorflow can be used on both CPU & GPU where inference (running pre-trained model) is recommended to run on the CPU (because of overhead of starting GPU sessions) while training on GPU will outperform the CPU depending on your hardware spec.

![](images/tensorflow_logo.png)

### Run Tensorflow Object Detection API Demo on CPU (inference)
First clone models:
```Bash
git clone https://github.com/tensorflow/models.git          [1]
```
in this folder. Start a jupyter notebook on CPU by:
```Bash
make jupyter
```
which starts up jupyter notebooks session in object-detection folder. You can access it via `http://127.0.0.1:8888` and a token which is shown in the shell. Now you are ready to run the demo example:
```Bash
./models/research/object_detection/object_detection_tutorial.ipynb
```
When successfull you will get something like:

![](images/tensorflow_jupyter_demo.png)

To stop the demo type `Ctrl-C`.

### Install NVIDIA docker
Check this [link](https://github.com/NVIDIA/nvidia-docker) on how to install *NVIDIA Container Runtime for Docker*.
If you don't have docker check this tutorial to install it on [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/).
When successfull this command will run
```Bash
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```
and give something like:
```Bash
Thu Nov 15 10:20:49 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.77                 Driver Version: 390.77                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 970     Off  | 00000000:03:00.0  On |                  N/A |
|  1%   46C    P8    12W / 200W |    380MiB /  4039MiB |      5%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
This image explains it all:
![](images/nvidia_docker.png)

### Specialized images for training
In order to do GPU accelerated calculations based on the NVIDIA cuDNN and tensorflow
use the image `heliconwave/object-detection`. You can build it via:
```Bash
make docker-image
```
To enter bash type:
```Bash
make docker-bash
```
or alternatively using `docker-compose`:
```
LOCAL_USER_ID=$(id -u ${USER}) DISPLAY=${DISPLAY} docker-compose run object-detection bash
```
If you are running object-detection API from `[1]` you need run this command before outside the docker container:
```
protoc object_detection/protos/*.proto --python_out=.             [2]
```
To be able to run X-server applications like opencv from Docker you need to type on your host OS:
```
xhost +local:docker                                               [3]
```
where docker is the user for your docker deamon, which should not be root following best practise.


### Examples
Please check the [`TRAINING.md`](TRAINING.md) on how to start using the docker image for training. 

#### GPU
First make sure you have a NVIDIA graphic card to be able to run these examples:
```
lspci | grep -i nvidia
```

Run git clone on `[1]` inside this project. It will be ignored by git. Run command `[2]` on models you just cloned and then allow X server via docker with `[3]`. Copy `test.mp4` and `test_object_detection.py` from `scripts` to `models/research/object_detection/`. Now start the demo by
```Bash
make docker-bash
```
or use docker-compose:
```Bash
LOCAL_USER_ID=$(id -u ${USER}) DISPLAY=${DISPLAY} docker-compose run object-detection bash
```
and change folder to
```Bash
cd /tmp/models/research/object_detection/
```
and start the demo:
```Bash
python test_object_detection.py
```
Feel free to modify the script `test_object_detection.py` to use other models or your own models and also to use other assets. If you have a webcamera the simple demo will display the video from the web-camera with overlays of object detections. Otherwise a test asset will be used.
Quit the example by typing q.

#### CPU
Like for GPU you need first download the `[1]` inside this project. Then copy
`test_object_detection.py` from `scripts` to `models/research/object_detection/`. Now create your python environment:
```
make venv
```
which downloads Tensorflow for CPU. Activate your new environment:
```
source ./venv/bin/activate
```
and cd directory to `models/research/object_detection/`. Run the demo by:
```
python test_object_detection.py
```
You will see a video of yourself if you have a webcamera otherwise a short clip. Quit the example by typing q.

### Custom models
If you need to run a custom model that you trained you can copy it to `./models/research/object-detection/data` and the model folder need to be stored under `./models/research/object-detection`. To run your new model you need to edit the python script `test_object_detection.py`. These are the minimum changes required to run a custom model:

Change line 35:
```
return False
```
Change line 37:
```
video_files = ['my_eval_asset.mp4']
```
Change `MODEL_NAME` line 56:
```
MODEL_NAME = 'my_custom_model'
```
Change model line 67:
```
PATH_TO_LABELS = os.path.join('data', 'my-object-detection-model.pbtxt')
```
Run the specific model:
```
> python test_object_detection.py
```
## Movidius
Check [README](docker/ncsdk/README.md) for more information.