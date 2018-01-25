import numpy as np
from struct import unpack

with open("bathtub_0107_4_1_000.octree", "rb") as binary_file:
    intsize = 4
    floatsize = 4
    endian = 'little'
    binary_file.seek(0)  # Go to beginning
    total_nodes_bytes = binary_file.read(intsize)
    total_nodes = int.from_bytes(total_nodes_bytes, byteorder=endian)
    print(total_nodes)

    final_nodes_bytes = binary_file.read(intsize)
    final_nodes = int.from_bytes(final_nodes_bytes, byteorder=endian)
    print(final_nodes)

    depth_bytes = binary_file.read(intsize)
    depth = int.from_bytes(depth_bytes, byteorder=endian)
    print(depth)

    full_layer_bytes = binary_file.read(intsize)
    full_layer = int.from_bytes(full_layer_bytes, byteorder=endian)
    print(full_layer)

    node_num_data = np.empty(depth + 1, dtype=int)
    for i in range(depth + 1):
        node_num_data_bytes = binary_file.read(intsize)
        node_num_data[i] = int.from_bytes(
            node_num_data_bytes, byteorder=endian)
    print(node_num_data)

    node_num_accu = np.empty(depth + 2, dtype=int)
    for i in range(depth + 2):
        node_num_accu_bytes = binary_file.read(intsize)
        node_num_accu[i] = int.from_bytes(
            node_num_accu_bytes, byteorder=endian)
    print(node_num_accu)

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
    if len(label_data_bytes) != 0:
        labels_chunked = [label_data_bytes[i:i + intsize]
                          for i in range(0, len(label_data_bytes), intsize)]
        label_data = map(lambda x: int.from_bytes(
            x, byteorder=endian), labels_chunked)
