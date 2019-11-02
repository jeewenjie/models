import tensorflow as tf

from research.object_detection.utils import dataset_util

import os
import hashlib

import cv2


#img = cv2.imread('path/to/img',0)
#height, width = img.shape[:2]

flags = tf.compat.v1.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('data_root','','Root of data directory')
flags.DEFINE_string('dataset','','Name of dataset')
FLAGS = flags.FLAGS

dict = {0:'mk',
        1:'gucci',
        2:'chanel',
        3:'supreme',
        4:'adidas0',
        5:'puma',
        6:'nike',
        7:'prada',
        8:'lacoste',
        9:'hh'}

def create_tf_example(line):
  # TODO(user): Populate the following variables from your example.

  #filename = None # Filename of the image. Empty if image is not from file
  #encoded_image_data = None # Encoded image bytes

  image_format = b'jpeg' # b'jpeg' or b'png'

  elements = line.split(' ')

  encoded_image_data = open(os.path.join(FLAGS.data_root,elements[0]), 'rb').read()
  filename = elements[0].split('/')[-1].replace('.jpg','').encode('utf-8')
  source_id = filename
  key = hashlib.sha256(encoded_image_data).hexdigest().encode('utf8')

  xmins = []
  xmaxs = []

  ymins = []
  ymaxs = []

  classes_text = []
  classes = []

  #TODO: Find way to convert cv2 image to bytes so won't open image twice.
  img = cv2.imread(os.path.join(FLAGS.data_root,elements[0]),0)
  height, width = img.shape[:2]

  for i in elements[1:]:

      _i = i.split(',')

      _xmins = int(_i[0])/width
      _xmaxs = int(_i[2])/width
      _ymins = int(_i[1])/height
      _ymaxs = int(_i[3])/height

      xmins.append(_xmins)
      xmaxs.append(_xmaxs)
      ymins.append(_ymins)
      ymaxs.append(_ymaxs)

      classes.append(int(_i[4]))
      classes_text.append(dict[int(_i[4])].encode('utf-8'))

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/key/sha256': dataset_util.bytes_feature(key)
  }))
  return tf_example


def main(_):
  writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)

  data_root = FLAGS.data_root
  dataset  = FLAGS.dataset

  annotation_f = open(os.path.join(data_root,dataset,'annotation.txt'),'r')
  for line in annotation_f:
      line = line.rstrip()
      tf_example = create_tf_example(line)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.compat.v1.app.run()
