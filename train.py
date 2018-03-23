from pyevtk.hl import pointsToVTK, imageToVTK
import tensorflow as tf
import numpy as np
import math
import struct
import os
import sys
from functools import partial

# Own ops
octree_conv_module = tf.load_op_library('./octree_conv.so')
octree_conv = octree_conv_module.octree_conv

def to_point(depth, key):
    key = np.asscalar(np.uint32(key))
    k = key.to_bytes(4, byteorder='little')
    k = bytearray(k)
    [x, y, z, _] = struct.unpack('B'*len(k), k)
    return [x, y, z]

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)

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
    # dataset = dataset.shuffle(100)
    # dataset = dataset.batch(batch_size)

    test_dataset = tf.data.TFRecordDataset([test_records])
    test_dataset = test_dataset.map(partial(parse_function, num_cats))
    # test_dataset = test_dataset.shuffle(shuffle_size)
    # test_dataset = test_dataset.batch(64)

    # For testing ---------
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    total_nodes = next_element['total_nodes']
    total_nodes = tf.Print(total_nodes, [total_nodes], message='\ntotal_nodes: ')
    # total_nodes.eval()

    final_nodes = next_element['final_nodes']
    final_nodes = tf.Print(final_nodes, [final_nodes], message='\nfinal_nodes: ')
    # final_nodes.eval()

    depth = next_element['depth']
    depth = tf.Print(depth, [depth], message='\ndepth: ')
    # depth.eval()

    node_num_data = next_element['node_num_data']
    node_num_data = tf.Print(
        node_num_data, [node_num_data], message='\nnode_num_data: ', summarize=200)
    # node_num_data.eval()

    node_num_accu = next_element['node_num_accu']
    node_num_accu = tf.Print(
        node_num_accu, [node_num_accu], message='\nnode_num_accu: ', summarize=200)

    key_data = next_element['key_data']
    key_data_shape = tf.shape(key_data)
    #key_data = tf.Print(
    #    key_data, [key_data], message='key_data: ', summarize=200)

    final_keys = tf.slice(key_data, final_nodes, [-1])
    #final_keys = tf.Print(final_keys, [final_keys], message='final_keys: ')
    final_keys_shape = tf.shape(final_keys)

    data = tf.reshape(next_element['data'], [1, -1, 1])
    data_shape = tf.shape(data)

    children_data = next_element['children_data']
    children_data_shape = tf.shape(children_data)

    ## CONV OP
    W = weight_variable([3, 3, 3, 1, 3])
    result = octree_conv(data, W, final_nodes, key_data, children_data, node_num_data, depth, [1])
    result_shape = tf.shape(result)

    concated = tf.concat([total_nodes, final_nodes, depth,
                          node_num_data, node_num_accu], 0)

    #concated.eval()
    print("Evaluating")

    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("tf_logs", sess.graph)

    #depth_, final_keys_, _ = sess.run([depth, final_keys, concated])
    res, r, s, d, c, _ = sess.run([result_shape, final_keys_shape, key_data_shape, data_shape, children_data_shape, concated])
    print("result shape: %s" % res)
    print("final_keys: %s" % r)
    print("key_data: %s" % s)
    print("data: %s" % d)
    print("children data: %s" % c)

    print("Calculating result")
    # res = [to_point(depth_[0], k) for k in final_keys_]
    # res = list(zip(*res))
    # data = np.full(len(res[0]), 250, dtype=np.int32)

    # x = np.array(res[0], dtype=np.int32)
    # y = np.array(res[1], dtype=np.int32)
    # z = np.array(res[2], dtype=np.int32)

    # pointsToVTK("./points", x, y, z, data={"data": data})


def main():
    dir_name = sys.argv[1]
    train(dir_name)


if __name__ == "__main__":
    main()
