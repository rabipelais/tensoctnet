import tensorflow as tf

from tensorflow.python.framework import ops

# Own ops
# Conv
octree_conv_module = tf.load_op_library('./octree_conv.so')
octree_conv = octree_conv_module.octree_conv

octree_conv_grad_module = tf.load_op_library('./octree_conv_gradient.so')
octree_conv_grad = octree_conv_grad_module.octree_conv_gradient

# Pooling
octree_pooling_module = tf.load_op_library('./octree_pooling.so')
octree_pooling = octree_pooling_module.octree_pooling

octree_pooling_grad_module = tf.load_op_library('./octree_pooling_gradient.so')
octree_pooling_grad = octree_pooling_grad_module.octree_pooling_gradient

# Full layer
octree_full_layer_module = tf.load_op_library('./octree_full_layer.so')
octree_full_layer = octree_full_layer_module.octree_full_layer

octree_full_layer_grad_module = tf.load_op_library('./octree_full_layer_gradient.so')
octree_full_layer_grad = octree_full_layer_grad_module.octree_full_layer_gradient


# Gradients
@ops.RegisterGradient("OctreeConv")
def _octree_conv_grad(op, grad):
    original_input = op.inputs[0]
    kernel = op.inputs[1]
    final_nodes = op.inputs[2]
    key_data = op.inputs[3]
    children_data = op.inputs[4]
    node_num_data = op.inputs[5]
    current_depth = op.inputs[6]
    strides = op.inputs[7]

    [input_grad, kernel_grad] = octree_conv_grad(
        grad,
        kernel,
        final_nodes,
        key_data,
        children_data,
        node_num_data,
        current_depth,
        strides,
        original_input)

    return [input_grad, kernel_grad, None, None, None, None, None, None]


@ops.RegisterGradient("OctreePooling")
def _octree_pooling_grad(op, grad):
    original_input = op.inputs[0]
    final_nodes = op.inputs[1]
    key_data = op.inputs[2]
    children_data = op.inputs[3]
    node_num_data = op.inputs[4]
    current_depth = op.inputs[5]

    input_grad = octree_pooling_grad(
        grad,
        final_nodes,
        key_data,
        children_data,
        node_num_data,
        current_depth,
        original_input)

    return [input_grad, None, None, None, None, None]


@ops.RegisterGradient("OctreeFullLayer")
def _octree_full_layer_grad(op, grad):
    # original_input = op.inputs[0]
    node_num_data = op.inputs[1]
    current_depth = op.inputs[2]

    input_grad = octree_full_layer_grad(
        grad,
        node_num_data,
        current_depth)

    return [input_grad, None, None]
