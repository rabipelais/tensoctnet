import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from struct import unpack


def read_data(dir):
    labels_dict = {}
    current_label_idx = 0
    # Read the data
    for (dirpath, dirnames, filenames) in os.walk(dir):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for name in filenames:
            basename, ext = os.path.splitext(name)
            if ext != '.octree':
                continue

            components = basename.split('_')
            train_p = components[0]
            label_name = "-".join(components[1:-4])

            data = read_file(os.path.join(dirpath, name))

            label_idx = -1
            if label_name in labels_dict:
                label_idx = labels_dict[label_name]
            else:
                labels_dict[label_name] = current_label_idx
                label_idx = current_label_idx
                current_label_idx += 1

            if train_p == "train":
                train_data.append(data)
                train_labels.append(label_idx)
            else:
                test_data.append(data)
                test_labels.append(label_idx)

        train = zip(train_data, one_hot(train_labels, len(labels_dict)))
        test = zip(test_data, one_hot(test_labels, len(labels_dict)))

        return train, test, labels_dict


def read_file(filename):
    with open(filename, "rb") as binary_file:
        intsize = 4
        floatsize = 4
        endian = 'little'
        binary_file.seek(0)  # Go to beginning
        total_nodes_bytes = binary_file.read(intsize)
        total_nodes = int.from_bytes(total_nodes_bytes, byteorder=endian)
        # print(total_nodes)

        final_nodes_bytes = binary_file.read(intsize)
        final_nodes = int.from_bytes(final_nodes_bytes, byteorder=endian)
        # print(final_nodes)

        depth_bytes = binary_file.read(intsize)
        depth = int.from_bytes(depth_bytes, byteorder=endian)
        # print(depth)

        full_layer_bytes = binary_file.read(intsize)
        full_layer = int.from_bytes(full_layer_bytes, byteorder=endian)
        # print(full_layer)

        node_num_data = np.empty(depth + 1, dtype=int)
        for i in range(depth + 1):
            node_num_data_bytes = binary_file.read(intsize)
            node_num_data[i] = int.from_bytes(
                node_num_data_bytes, byteorder=endian)
        # print(node_num_data)

        node_num_accu = np.empty(depth + 2, dtype=int)
        for i in range(depth + 2):
            node_num_accu_bytes = binary_file.read(intsize)
            node_num_accu[i] = int.from_bytes(
                node_num_accu_bytes, byteorder=endian)
        # print(node_num_accu)

        key_data = np.empty(total_nodes, dtype=int)
        for i in range(total_nodes):
            key_data_bytes = binary_file.read(intsize)
            key_data[i] = int.from_bytes(
                key_data_bytes, byteorder=endian)
        # print(key_data)

        children_data = np.empty(total_nodes, dtype=int)
        for i in range(total_nodes):
            children_data_bytes = binary_file.read(intsize)
            children_data[i] = int.from_bytes(
                children_data_bytes, byteorder=endian)
        # print(children_data)

        data_data = np.empty(final_nodes * 3, dtype=float)
        for i in range(final_nodes * 3):
            data_data_bytes = binary_file.read(floatsize)
            tmp = unpack('f', data_data_bytes)
            data_data[i] = tmp[0]
        # print(data_data)

        label_data_bytes = binary_file.read()
        label_data = []
        if len(label_data_bytes) != 0:
            labels_chunked = [label_data_bytes[i:i + intsize]
                              for i in range(0, len(label_data_bytes), intsize)]
            label_data = map(lambda x: int.from_bytes(
                x, byteorder=endian), labels_chunked)

        res = {'total_nodes': total_nodes,
               'final_nodes': final_nodes,
               'depth': depth,
               'full_layer': full_layer,
               'node_num_data': node_num_data,
               'node_num_accu': node_num_accu,
               'key_data': key_data,
               'children_data': children_data,
               'data': data_data,
               'label_data': label_data}
        return res


def one_hot(indices, cats):
    one_hots = []
    for i in indices:
        hotty = np.zeros(cats, dtype=np.int64)
        hotty[i] = 1
        one_hots.append(hotty)
    return np.array(one_hots)


def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.trainBytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read the files in DIR and join them into TFRecords. It assumes that each filename has the following format: {test,train}_LABEL_number_depth_full-layer_view.vox, where LABEL is the category of the object, and number an arbitrary identifier. It will output one file for the test data, and one for the training data, and a text file with a label-class id correspondence.\n Example file name: test_bathtub_0229_4_2_003.octree')

    parser.add_argument('source', metavar='DIR',
                        help='Directory with the .octree files.')

    parser.add_argument('--destination', '-o',
                        help='Name of the output directory. If not given, if will output the files into the source dir.')

    args = parser.parse_args()

    dir_name = args.source

    if args.destination:
        out_dir_name = args.destination
    else:
        out_dir_name = dir_name

    train_set, test_set, labels_dict = read_data(dir_name)

    # Write training data
    train_writer = tf.python_io.TFRecordWriter(
        os.path.join(out_dir_name, "training.tfrecord"))
    for dic, label in train_set:
        example = tf.train.Example(features=tf.train.Features(feature={
            'total_nodes': _int64_feature(dic['total_nodes']),
            'final_nodes': _int64_feature(dic['final_nodes']),
            'depth': _int64_feature(dic['depth']),
            'full_layer': _int64_feature(dic['full_layer']),
            'node_num_data': _int64_feature_list(dic['node_num_data']),
            'node_num_accu': _int64_feature_list(dic['node_num_accu']),
            'key_data': _int64_feature_list(dic['node_num_data']),
            'children_data': _int64_feature_list(dic['children_data']),
            'data': _float_feature_list(dic['data']),
            'label_data': _int64_feature_list(dic['label_data']),
            'label_one_hot': _int64_feature_list(label)}))
        train_writer.write(example.SerializeToString())

    # Write test data
    test_writer = tf.python_io.TFRecordWriter(
        os.path.join(out_dir_name, "test.tfrecord"))
    for dic, label in test_set:
        example = tf.train.Example(features=tf.train.Features(feature={
            'total_nodes': _int64_feature(dic['total_nodes']),
            'final_nodes': _int64_feature(dic['final_nodes']),
            'depth': _int64_feature(dic['depth']),
            'full_layer': _int64_feature(dic['full_layer']),
            'node_num_data': _int64_feature_list(dic['node_num_data']),
            'node_num_accu': _int64_feature_list(dic['node_num_accu']),
            'key_data': _int64_feature_list(dic['node_num_data']),
            'children_data': _int64_feature_list(dic['children_data']),
            'data': _float_feature_list(dic['data']),
            'label_data': _int64_feature_list(dic['label_data']),
            'label_one_hot': _int64_feature_list(label)}))
        test_writer.write(example.SerializeToString())

    # for serialized_example in tf.python_io.tf_record_iterator(test_records):
    #    example = tf.train.Example()
    #    example.ParseFromString(serialized_example)
    #    height = np.array(example.features.feature['height'].int64_list.value)
    #   data = np.array(example.features.feature['data'].float_list.value)

    # Write labels dictionary
    labels_file = os.path.join(out_dir_name, "labels.txt")
    with open(labels_file, 'w') as file_handle:
        for cat in labels_dict:
            file_handle.write(cat + " ")
            file_handle.write(str(labels_dict[cat]))
            file_handle.write("\n")
