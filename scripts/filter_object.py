import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import os.path
import collections
import cv2
import argparse


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

def load_model():
  # What model to use / download.
  MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
  print("Using model: %s" % PATH_TO_CKPT)
  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
  NUM_CLASSES = 90
  ## Download Model
  if not os.path.isfile(PATH_TO_CKPT):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
  else:
    print("File exists: {0}".format(PATH_TO_CKPT))
  ## Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  return detection_graph, category_index


# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_display_string(
  boxes,
  classes,
  scores,
  category_index,
  instance_masks=None,
  instance_boundaries=None,
  keypoints=None,
  max_boxes_to_draw=20,
  min_score_thresh=.5,
  agnostic_mode=False,
  groundtruth_box_visualization_color='black',
  skip_scores=False,
  skip_labels=False):
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  strings = []
  objects = []
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_dic = {}
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_dic[str(class_name)] = format(int(100*scores[i]))
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_dic[display_str] = format(int(100*scores[i]))
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            strings.append(display_str)
            objects.append(display_dic)
  return objects



# Detection
def run_detection(video_files, detection_graph, category_index):
  for videofile in video_files:
    print("processing file {0}".format(videofile))
  
    try:
      cap = cv2.VideoCapture(videofile)
    except:
      print("Could not open video file {0}".format(videofile))
  
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        objects = []
        while True:
          ret, image_np = cap.read()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          try: 
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
          except:
            print("got exception continue...")
            break
          # Visualization of the results of a detection.
          display = get_display_string(
            np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
            )
          if display:
            print("scores: {0}".format(display))
  
          cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
          if cv2.waitKey(25) & 0xFF == ord('n'):
            cv2.destroyAllWindows()
            break
          elif cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()
            break
          if display:
            objects.append(display)
        return objects


def parse_arguments(lista):
  parser = argparse.ArgumentParser(description='Object boolean filter to find objects in video files.')
  parser.add_argument(
    'input',
    help='a path to a video file')
  parser.add_argument(
    '--model',
    dest='model',
    help='a path to the pb model file; the default is `ssd_mobilenet_v1_coco_11_06_2017`')
  parser.add_argument(
    '--silence-threshold',
    dest='silence_threshold',
    type=float,
    help=("indicates what sample value you should treat as silence; "
           "the default is `0.5`"))
  parser.add_argument(
    '--silence-min-duration',
    dest='silence_min_duration_sec',
    type=float,
    help=("specifies a period of silence that must exist before video is "
          "not copied any more; the default is `0.1`"))
  parser.add_argument(
    '--verbose',
    dest='verbose',
    action='store_true',
    help='print more logs')

  parser.set_defaults(
    model='ssd_mobilenet_v1_coco_11_06_2017',
    silence_min_duration_sec=0.1,
    silence_threshold=0.5,
    verbose=False)

  args = parser.parse_args(lista)
  # Video assets to stream
  video_files = ['test.mp4']
  detection_graph, category_index = load_model()
  objects = run_detection([args.input], detection_graph, category_index)
  if objects:
    print("objects: {0}".format(objects))
    return objects


def main():
  parse_arguments(sys.argv[1:])

if __name__ == "__main__":
  main()
