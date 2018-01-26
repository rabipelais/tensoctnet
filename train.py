import tensorflow as tf
import numpy as np
import math
import os
import sys
from functools import partial


def parse_function(cats, record):
    features = {
        'total_nodes': tf.FixedLenFeature((1), tf.int64),
        'final_nodes': tf.FixedLenFeature((1), tf.int64),
        'depth': tf.FixedLenFeature((1), tf.int64),
        'full_layer': tf.FixedLenFeature((1), tf.int64),
        'node_num_data': tf.VarLenFeature(tf.int64),
        'node_num_accu': tf.VarLenFeature(tf.int64),
        'key_data': tf.VarLenFeature(tf.int64),
        'children_data': tf.VarLenFeature(tf.int64),
        'data': tf.VarLenFeature(tf.float32),
        'label_data': tf.VarLenFeature(tf.int64),
        'label_one_hot': tf.FixedLenFeature((cats), tf.int64)
    }

    pf = tf.parse_single_example(record, features)  # Parsed features

    total_nodes = tf.cast(pf['total_nodes'], tf.int64)
    final_nodes = tf.cast(pf['final_nodes'], tf.int64)
    depth = tf.cast(pf['depth'], tf.int64)
    full_layer = tf.cast(pf['full_layer'], tf.int64)

    node_num_data = pf['node_num_data'].values
    # node_num_data.set_shape(depth_)
    node_num_accu = pf['node_num_accu'].values
    # node_num_accu.set_shape((depth[0] + 2))
    key_data = pf['key_data'].values
    # key_data.set_shape((total_nodes[0]))
    children_data = pf['children_data'].values
    # children_data.set_shape((total_nodes[0]))

    data = pf['data'].values
    # data.set_shape((final_nodes[0] * 3))

    label_data = pf['label_data'].values

    res = {'total_nodes': total_nodes,
           'final_nodes': final_nodes,
           'depth': depth,
           'full_layer': full_layer,
           'node_num_data': node_num_data,
           'node_num_accu': node_num_accu,
           'key_data': key_data,
           'children_data': children_data,
           'data': data,
           'label_data': label_data,
           'cat_one_hot': pf['label_one_hot']}

    return res


def train(dir_name):
    sess = tf.InteractiveSession()

    training_records = os.path.join(dir_name, "training.tfrecord")
    test_records = os.path.join(dir_name, "test.tfrecord")

    num_cats = sum(1 for line in open(os.path.join(dir_name, "labels.txt")))
    print("Found " + str(num_cats) + " categories")

    dataset = tf.data.TFRecordDataset([training_records])
    dataset = dataset.map(partial(parse_function, num_cats))
    # dataset = dataset.shuffle(shuffle_size)
    # dataset = dataset.batch(batch_size)

    test_dataset = tf.data.TFRecordDataset([test_records])
    test_dataset = test_dataset.map(partial(parse_function, num_cats))
    # test_dataset = test_dataset.shuffle(shuffle_size)
    # test_dataset = test_dataset.batch(64)

    print(dataset.output_shapes)


def main():
    dir_name = sys.argv[1]
    train(dir_name)


if __name__ == "__main__":
    main()
