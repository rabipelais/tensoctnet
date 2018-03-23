#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("OctreeConv")
    .Input("input: T")
    .Input("kernel: T")
    .Input("final_nodes: int64")
    .Input("key_data: int64")
    .Input("children_data: int64")
    .Input("node_num_data: int64")
    .Input("current_depth: int64")
    .Input("strides: int64")
    .Output("output: T")
    .Attr("T: {float} = DT_FLOAT");

template <typename T>
class OctreeConvOp : public OpKernel {
 public:
	explicit OctreeConvOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		//Grab the input tensors

		// This should have the following dimensions?
		// [ batch, data_dim, in_depth ]
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();

		// Input filter is of the following dimensions:
		// [ filter_x, filter_y, filter_z, in_depth, out_depth]
		const Tensor& kernel_tensor = context->input(1);
		auto kernel = kernel_tensor.flat<T>();

		OP_REQUIRES(context, input_tensor.dims() == 3,
					errors::InvalidArgument("input must be 3-dimensional",
										   input_tensor.shape().DebugString()));
		OP_REQUIRES(context, kernel_tensor.dims() == 5,
					errors::InvalidArgument("kernel must be 5-dimensional: ",
										   kernel_tensor.shape().DebugString()));

		// The last dimension for input is in_depth. It must be the same as the
		// filter's in_depth.
		const int64 in_depth = input_tensor.dim_size(2);

		OP_REQUIRES(context, in_depth == kernel_tensor.dim_size(3),
					errors::InvalidArgument(
						"input and kernel must have the same depth: ", in_depth,
						" vs ", kernel_tensor.dim_size(3)));

		// The last dimension for filter is out_depth.
		const int out_depth = static_cast<int>(kernel_tensor.dim_size(3));


		const Tensor& final_nodes_tensor = context->input(2);
		auto final_nodes = final_nodes_tensor.flat<int64>()(0);

		const Tensor& children_data_tensor = context->input(4);
		auto children_data = children_data_tensor.flat<int64>();

		const Tensor& key_data_tensor = context->input(3);
		auto key_data = key_data_tensor.flat<int64>();

		const Tensor& node_num_data_tensor = context->input(5);
		auto node_num_data = node_num_data_tensor.flat<int64>();

		const Tensor& current_depth_tensor = context->input(6);
		auto current_depth = current_depth_tensor.flat<int64>()(0);

		OP_REQUIRES(context, input_tensor.dim_size(1) == node_num_data(current_depth) * 3,
					errors::InvalidArgument(
						"input data must have node_num_data[current_depth] * 3 entries: ", input_tensor.dim_size(1),
						" vs ", node_num_data(current_depth) * 3));

		const Tensor& strides_ = context->input(7);
		auto stride = strides_.flat<int64>()(0);
	    OP_REQUIRES(context, strides_.dims() == 1,
					errors::InvalidArgument("Sliding window strides field must "
"specify just 1 dimension."));
		OP_REQUIRES(context, stride == 1 || stride == 2,
					errors::InvalidArgument("Stride must currently be only 1 or 2: ", stride));

		// Calculate output shape
		auto out_size = stride == 1? input_tensor.dim_size(1) : node_num_data(current_depth - 1) * 3; //TODO check for root

		TensorShape output_shape;
		output_shape.AddDim(input_tensor.dim_size(0));
		output_shape.AddDim(out_size);
		output_shape.AddDim(out_depth);

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
		auto output = output_tensor->flat<T>();

		// Precalculate some neighbourhood info for 3-kernels
		int ni3[256];
		int id = 0;
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				for (int k = 0; k < 2; ++k) {
					for (int x = 0; x < 3; ++x) {
						for (int y = 0; y < 3; ++y) {
							for (int z = 0; z < 3; ++z) {
								ni3[id++] = (x + i << 4) | (y + j << 2) | z + k;

		// octree2col
		const int octree_h = current_depth << 3 * (stride - 1); // = `div` 8
		const int kernel_size = kernel_tensor.dim_size(0) * kernel_tensor.dim_size(1) * kernel_tensor.dim_size(2); // only tested for 3 * 3 * 3, should probably check for this


		for(int c = 0; c < in_depth; ++c) {
			for(int k = 0; k < kernel; ++k) {
				for(int h = 0; h < current_depth; ++h) {

					const int index = stride == 2 ? (h << 6) + ni[k] :
						(h >> 3 << 6) + ni[(h % 8) * kernel + k];

					const int p = neigh[index];

					data_col[(c*kernel + k)*height + h] = p == -1 ?
						Dtype(0) : data_octree[c*octree_h + p];
				}
			}
		}


	}
};

REGISTER_KERNEL_BUILDER(
    Name("OctreeConv")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    OctreeConvOp<float>);
