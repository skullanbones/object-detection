# Object detection
How to train, run and understand object detection on both CPU & GPU using TensorFlow and Docker. Docker is used instead of installing CUDA on your machine. It simplifies the dependencies dramatically and you can be up and runing within minutes instead of hours.

## Table of contents
1. [Acronyms](##Acronyms)
2. [Assumptions](##Assumptions)
3. [Goal](##Goal)
4. [Requirements SW/HW](##Requirements%20SW/HW)
5. [Download the repo](##Download%20the%20repo)
6. [Download models](##Download%20models)
7. [Copy python script to research folder](##Copy%20python%20script%20to%20research%20folder)
8. [Prepare protobuffers](##Prepare%20protobuffers)
9. [Example: Run default model on CPU](##Example:%20Run%20default%20model%20on%20CPU)
10. [Example: Run default model on GPU](##Example:%20Run%20default%20model%20on%20GPU)
11. [Using LabelImg](##Using%20LabelImg)
12. [Prepare for training on labeled images](##Prepare%20for%20training%20on%20labeled%20images)
13. [Download pre-trained models](##Download%20pre-trained%20models)
14. [Actual training on labeled images](##Actual%20training%20on%20labeled%20images)
15. [Create inference graph and run trained model](##Create%20inference%20graph%20and%20run%20trained%20model)
16. [References](##References)


## Acronyms

| Acronym                          | Meaning                     |
|----------------------------------|-----------------------------|
| CUDA                             | Compute Unified Device Architecture
| cuDNN                            | CUDA® Deep Neural Network library
| GPU                              | Graphics Processing Unit
| CPU                              | Central Processing Unit
| SW                               | Software
| HW                               | Hardware
| ML                               | Machine Learning


## Assumptions
If no path is given you should then be in
```
models/research/object_detection
```
This is our default working directory.

## Goal
To train and run custom object detection models with AI assistance using Google TensorFlow.

## Requirements SW/HW
Hardware requirement is a PC / laptop with a NVIDIA GPU and Intel CPU. The GPU should support CUDA 10. 

| Requirement                          | Command                     |
|----------------------------------|-----------------------------|
| Ubuntu 18.04                     | lsb_release -a
| NVIDIA Docker version > v2       | nvidia-docker version
| Docker > v18                     | docker --version
| Protocol buffers: protoc > v3.6  | protoc --version
| Virtualenv > v15                 | virtualenv --version
| At least nvidia-driver-430       | nvidia-smi

In the top right corner see example below you can find the CUDA and driver version. For example:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.26       Driver Version: 430.26       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:05:00.0  On |                  N/A |
| 42%   72C    P2    97W / 250W |  11023MiB / 11175MiB |     23%      Default |
```



## Download the repo
```
$ git clone git@github.com:skullanbones/object-detection.git                [0]
```

## Download models
Step into [0](##Download%20the%20repo):
```
$ cd object-detection
```
and clone models:
```
$ git clone https://github.com/tensorflow/models.git
```
## Copy python script to research folder
```
$ cd ..
$ cp scripts/test_object_detection.py models/research/object_detection/
```
Copy asset:
```
$ cp scripts/test.mp4 models/research/object_detection/
```

## Prepare protobuffers
```
$ cd models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ cd ../..
```

## Example: Run default model on CPU
You need to be in the root folder object-detection before running these commands.
**[1]**:
```
$ make venv
$ source ./venv/bin/activate
$ cd models/research/object_detection/
$ python3 test_object_detection.py
```

Press q to quit

## Example: Run default model on GPU
You need to be in the root folder object-detection before running these commands.
Add X server to docker user
**[2]**:
```
$ xhost +local:docker

$ make docker-bash
$ cd /tmp/models/research/object_detection/
$ python3 test_object_detection.py
```

Press q to quit

## Using LabelImg
First clone the repo:
```
$ git clone https://github.com/tzutalin/labelImg.git
```

```
$ cd labelImg/
$ virtualenv -p python3 venv
$ source ./venv/bin/activate
$ pip3 install -r requirements/requirements-linux-python3.txt
$ make qt5py3
```

Start the program by:
```
$ python3 labelImg.py
$ python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

The last command is if you have a folder with images you want to label and a classification file with your labels you want to use and label with.

## Prepare for training on labeled images
A good [youtube tutorial](https://www.youtube.com/watch?v=kq2Gjv_pPe8) explaining this step in detail
Create your image folder and lable all images you want to use.
```
$ mkdir target_labeled images
```

Create evaluation (test) and training directories:
```
$ mkdir images/test
$ mkdir images/train
```

Move 10 % of the images + xml to `test/` and the rest to `train/` folders as the youtube clip suggest.

Download 2 files `xml_to_csv.py` and `generate_tfrecord.py` [from](https://github.com/datitran/raccoon_dataset).

And copy them to `models/research/object_detection`.

Add a line to 28 in file `xml_to_csv.py`:

```
for directory in ['train', 'test']:
```

To loop over both the train and test directories. Next change the line32 from

```
xml_df.to_csv('raccoon_labels.csv', index=None)
```

to:

```
xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
```

Change line 30 to:
```
image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
```

Place the `xml_to_csv.py` file in `models/research/object_detection` directory and run it :
```
$ python3 xml_to_csv.py
```

Remember to source your env in [1](##Example:%20Run%20default%20model%20on%20CPU) before running python.

You should now have 2 new files in the data/ directory:
```
-rw-r--r-- 1 guest guest   48 jun 27 18:11 train_labels.csv
-rw-r--r-- 1 guest guest   48 jun 27 18:11 test_labels.csv
```

Please check these 2 files are not empty but have similar amount of lines as the number of images per directory (train and test).

Change the file `generate_tfrecord.py` on the following lines:
Line 32:

Change to your first classifier and if you have more than 1 add them here and return the new number starting from 1 up with +1 for each new classification. Note you will get errors if these labels don't match your labeled images. For example:

```
def class_text_to_int(row_label):
   if row_label == 'bearded_collie':
       return 1
   if row_label == 'person':
       return 2
   else:
       None
```

for a 2 classification training with “bearded_collie” and “person”.

You may also need change line 20:
```
from object_detection.utils import dataset_util
```
to
```
from utils import dataset_util
```

Now in order to generate record files run for each image sub directory:
```
$ python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/

$ python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train/ 
```

## Download pre-trained models

Please check [this](https://www.youtube.com/watch?v=JR8CmWyh2E8&t=3s) video tutorial for more detailed information on how to train using an existing model.

Decide on a starting model. In the example mobilenet was picked for speed. But depending on the application another model might be more suitable. It is also possible to create your own model but that is beyond this article. This is left as a bonus assignment.

Copy the content from this file on the internet to a file with the same name: https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config

Do the changes as in the video.For example on line 2
```
num_classes: 2
```

Line 156:
```
fine_tune_checkpoint: "ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
```

Line 176:
```
input_path: "data/train.record"
```

Line 178 and 190:
```
label_map_path: "data/object-detection.pbtxt"
```

Line 188:
```
input_path: "data/test.record"
```

Download the model `ssd_mobilenet_v1_coco` [from](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz). Extract this model in models/research/object-detection:
```
$ tar xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

Create a directory called `training` and in this directory create a new file called `object-detection.pbtxt`. Paste this into this file:
```
$ mkdir training
```

```
item {
    id: 1
    name: 'bearded_collie'
}
item {
    id: 2
    name: 'person'
}
```

If you have other classifiers and another amount you need to reflect that here in this file.

Now you need to run the command so that python can find the slim directory. Note run this command from models/research folder.
```
$ export PYTHONPATH=`pwd`:`pwd`/slim
```

If this doesn't work which we will see at a later step there is an alternative involving in copy the `slim/nets` and `slim/deployment` to `models/research/object_detection/legacy` folder. Skip this step for now.



## Actual training on labeled images
Now you are ready to train with this command:

This can be done both on CPU & GPU but GPU is recommended since the time will vary hugely. For example on CPU the same model might take 5 hours but on a GPU it will take 1 hour instead (depending on the GPU type). Did you remember how to start a GPU sessions? If not please refer to [2](##Example:%20Run%20default%20model%20on%20GPU). Note that once inside Docker some commands need to be rerun like
```
$ export PYTHONPATH=`pwd`:`pwd`/slim
```
When you are in `/tmp/models/research` folder.
```
$ python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

If you run into memory exhaustion then you might need to lower the `batch_size` in `training/ssd_mobilenet_v1_pets.config`. Try half first and if that is not enough try half again. 

There are several parameters that are important for determining when a model is getting good. One is the loss parameter. The goal should be to minimize this at least below 1. One way of determining the loss parameter is from tensorboard which is a panel with all important metadata of each iteration during the process:
```
http://localhost:6006
```

If you cannot reach or converge to 1, try to figure out the reason. It should be possible… Please check this [link](https://www.youtube.com/watch?v=srPndLNMMpk&t=2s) on more information.

## Create inference graph and run trained model
Create a model directory that will be the output for your new model:
```
$ mkdir my_new_model_2019                            [4]
```
for example.

Now create the inference graph from your training:
```
$ python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-33964 \
    --output_directory my_new_model_2019
```

Where `--trained_checkpoint_prefix` should be your specific checkpoint. Check your training directory!

After that you are ready to test your newly created model! :) Do you remember how to do it?

Simply type:
```
$ python3 test_object_detection.py
```
But remember to change `MODEL_NAME` in the script to correspond to [4].



## References
1. [CUDA support](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements)
2. [cuDNN](https://developer.nvidia.com/cudnn)
