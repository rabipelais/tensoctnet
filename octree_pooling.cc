#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("OctreePooling")
    .Input("input: T")
    .Input("final_nodes: int64")
    .Input("key_data: int64")
    .Input("children_data: int64")
    .Input("node_num_data: int64")
    .Input("current_depth: int64")
    .Output("output: T")
    .Attr("T: {float} = DT_FLOAT");

template <typename T>
class OctreePoolingOp : public OpKernel {
 public:
	explicit OctreePoolingOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		//Grab the input tensors

		// This should have the following dimensions?
		// [ batch, data_dim, in_depth ]
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();

		OP_REQUIRES(context, input_tensor.dims() == 3,
					errors::InvalidArgument("input must be 3-dimensional",
										   input_tensor.shape().DebugString()));

		const int in_size = input_tensor.dim_size(1);
		// The last dimension for input is in_depth. It must be the same as the
		// filter's in_depth.
		const int64 in_depth = input_tensor.dim_size(2);

		// The last dimension for filter is out_depth.
		const int out_depth = in_depth;


		const Tensor& final_nodes_tensor = context->input(1);
		auto final_nodes = final_nodes_tensor.flat<int64>()(0);

		const Tensor& children_data_tensor = context->input(3);
		auto labels = children_data_tensor.flat<int64>();

		const Tensor& key_data_tensor = context->input(2);
		auto key_data = key_data_tensor.flat<int64>();

		const Tensor& node_num_data_tensor = context->input(4);
		auto node_num_data = node_num_data_tensor.flat<int64>();

		const Tensor& current_depth_tensor = context->input(5);
		auto current_depth = current_depth_tensor.flat<int64>()(0);

		OP_REQUIRES(context, input_tensor.dim_size(1) == node_num_data(current_depth) * 3,
					errors::InvalidArgument(
						"input data must have node_num_data[current_depth] * 3 entries: ", input_tensor.dim_size(1),
						" vs ", node_num_data(current_depth) * 3));

		// Calculate output shape
		auto out_size = node_num_data(current_depth - 1) * 3; //TODO check for root


		// Do the pooling. Var names correspond to OCNN. They're weird
		int channel = in_depth;
		int bottom_h = in_size;
		int top_h = in_size >> 3;

		std::vector<float> output_buffer(channel * in_size >> 3);

		for (int c = 0; c < channel; ++c) {
			for (int h = 0; h < top_h; ++h) {

				int hb = 8 * h;
				output_buffer[c * top_h + h] = input(c * bottom_h + hb);

				for (int idx = hb + 1; idx < hb + 8; ++idx) {
					if (input(c * bottom_h + idx) > output_buffer[c * top_h + h]) {
						output_buffer[c * top_h + h] = input(c * bottom_h + idx);
					}
				}
			}
		}

		TensorShape output_shape;
		output_shape.AddDim(input_tensor.dim_size(0));
		output_shape.AddDim(out_size);
		output_shape.AddDim(out_depth);

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
		auto output = output_tensor->flat<float>();

		//Recreate accum for nodes (why not just pass it?)
		std::vector<int> nodes_acc(node_num_data.size() + 1);
		nodes_acc[0] = 0;
		for(int i = 0; i < node_num_data.size(); i++) {
			nodes_acc[i+1] = nodes_acc[i] + node_num_data(i);
		}

		// Padding to account for empty nodes.
		for (int c = 0; c < channel; ++c)
		{
			for (int h = 0; h < out_size; ++h)
			{
				const long GUARD = 4294967295;
				output(c*out_size + h) = labels(nodes_acc[current_depth - 1] + h) == GUARD ? 0 : output_buffer[c*top_h + labels(nodes_acc[current_depth - 1] + h)];
			}
		}

	}
};

REGISTER_KERNEL_BUILDER(
    Name("OctreePooling")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    OctreePoolingOp<float>);
