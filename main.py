import numpy as np
import os
import tarfile
import six.moves.urllib as urllib
import tensorflow as tf
import zipfile
import cv2
from PIL import Image

from collections import defaultdict
from io import StringIO
import tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#%%
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#%%
# transfer image to array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

TEST_IMAGE_PATHS = ['test_images/image1.jpg', 'test_images/image2.jpg', 'test_images/image3.jpg', 'test_images/image4.jpg']

#%%
''' Load Pre-trained Model '''

PATH_TO_LIB = '/home/tiahu/tensorflow/models/research/object_detection'

MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(PATH_TO_LIB, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

#%%
''' Download/Unzip Model '''
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

#%%
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

#%%
''' Load Classification Labels '''
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#%%
''' =============================================== '''
''' ================= Kernel Code ================= '''
''' =============================================== '''

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        TV_BOX = np.array([[0, 0, 0, 0]])

        for image_path in TEST_IMAGE_PATHS:

            image = Image.open(image_path)  # LOAD IMAGES
            (width, height) = image.size

            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            box = np.squeeze(boxes)
            klass = np.squeeze(classes)
            score = np.squeeze(scores)

            tv_index = np.where(klass == 72)
            tv_box_raw = box[tv_index]
            tv_box = tv_box_raw.copy()

            for i in range(len(tv_box_raw)):
                tv_box[i, 0] = int(box[i, 0]*height)
                tv_box[i, 1] = int(box[i, 1]*width)
                tv_box[i, 2] = int(box[i, 2]*height)
                tv_box[i, 3] = int(box[i, 3]*width)

            tv_box_out = np.concatenate((TV_BOX, tv_box), axis=0)

            print(tv_box, '\n')
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow(image_path, cv2.resize(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), (1080, 720)))

print('Output\n', tv_box_out[1:])

cv2.waitKey(0)
cv2.destroyAllWindows()
