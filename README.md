# Computer vision with AI

## Tensorflow object detection
![](images/tensorflow_logo.png)

### Run Tensorflow Object Detection API Demo
First clone models:
```Bash
git clone https://github.com/tensorflow/models.git          [1]
```
in this folder which is mounted into docker when running the tensorflow GPU container by:
```Bash
make docker-jupyter
```
which starts up jupyter notebooks session in object-detection folder. You can access it via `http://127.0.0.1:8888` and a token which is shown in the shell. Now you are ready to run the demo example:
```Bash
./models/research/object_detection/object_detection_tutorial.ipynb
```
When successfull you will get something like:

![](images/tensorflow_jupyter_demo.png)

Top stop the container  you need to type in another shell:
```Bash
make docker-stop
```

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

### Specialized images
In order to do GPU accelerated calculations based on the NVIDIA cuDNN and tensorflow
use the image `heliconwave/object-detection`. You can build it via:
```Bash
make docker-image
```
To enter bash type:
```Bash
make docker-bash
```
If you are running object-detection API from `[1]` you need run this command before outside the docker container:
```
protoc object_detection/protos/*.proto --python_out=.             [2]
```
To be able to run X-server applications like opencv uses you need to type on your host OS:
```
xhost +local:docker                                               [3]
```
where docker is the user for your docker deamon, which should not be root following best practise.


### Examples
#### GPU
First make sure you have a NVIDIA graphic card to be able to run these examples:
```
lspci | grep -i nvidia
```

Run git clone on `[1]` inside this project. It will be ignored by git. Run command `[2]` on models you just cloned and then allow X server via docker with `[3]`. Copy `test.mp4` and `test_object_detection.py` from `scripts` to `models/research/object_detection/`. Now start the demo by
```
make docker-bash
```
and change folder to
```
cd /tmp/models/research/object_detection/
```
and start the demo:
```
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


## Movidius
![](images/movidius_logo.png)

Note only Movidius Neural Stick ver 1 works with NCSDK. To use the Neural Compute Stick ver2 OpenVINO is required and will not work with NCSDK. Docker images described below is for NVSDK ver 1.0 API.

### Docker
Build image:
```Bash
make docker-image-ncsdk
```
Run image with priviliged rights:
```Bash
make docker-bash-ncsdk
```

### Example
For all examples you need to download the App Zoo and start a docker container:
```Bash
git clone https://github.com/movidius/ncappzoo.git
make docker-bash-ncsdk
```
![](images/example1.png)

To run image classifier:
```Bash
cd ncappzoo/apps/object-detector
sudo python3 object-detector.py --image ../../data/images/pic_075.jpg 
```
To run live object-detection:
```Bash
cd ncappzoo/apps/live-object-detector
sudo make run
```

Verified working examples:
* security-cam
* video_face_matcher
* MultiStick_TF_Inception
* image-classifier
* rapid-image-classifier

Verified does NOT work
* gender_age_lbp
* live-image-classifier