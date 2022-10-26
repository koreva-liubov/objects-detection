import six
import requests
import os
import cv2
import pathlib
import PIL.Image as Image
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

objects_set = set()


def load_coco_model():
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_name = 'ssd_inception_v2_coco_2017_11_17'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname = model_name,
                                        origin = base_url + model_file,
                                        untar = True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    return category_index, str(model_dir)


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def get_classes_name_and_scores(
    boxes,
    classes,
    scores,
    category_index,
    max_boxes_to_draw=20,
    min_score_thresh=.8): # returns bigger than 80% precision
    display_str = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            if classes[i] in six.viewkeys(category_index):
                if (category_index[classes[i]]['name'] not in objects_set):
                    telegram_bot_sendtext(category_index[classes[i]]['name'] + " detected!")
                    print("Adding ", category_index[classes[i]]['name'], " to the set")
                    objects_set.add(category_index[classes[i]]['name'])
                    display_str['name'] = category_index[classes[i]]['name']
                    display_str['score'] = '{}%'.format(int(100 * scores[i]))

    if display_str:
        print(display_str)


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))


  # Actual detection.
    
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  # display(Image.fromarray(image_np))
  print("AA")
  cv2.imshow('object detection', image_np)
  print("BB")
  cv2.waitKey(0)
  print("CC")
  #cv2.destroyAllWindows()


category_index, model_str = load_coco_model()
detection_model = tf.saved_model.load(model_str)

#show_inference(detection_model, "models/research/object_detection/test_images/image1.jpg")
show_inference(detection_model, "muffins.jpg")
