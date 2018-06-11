#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("OctreeFullLayerGradient")
    .Input("input: T")
    .Input("node_num_data: int64")
    .Input("current_depth: int64")
    .Output("output: T")
    .Attr("T: {float} = DT_FLOAT");

template <typename T>
class OctreeFullLayerGradientOp : public OpKernel {
 public:
	explicit OctreeFullLayerGradientOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		//Grab the input tensors

		// This should have the following dimensions?
		// [ batch, data_dim, in_depth ]
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();

		OP_REQUIRES(context, input_tensor.dims() == 5,
					errors::InvalidArgument("input must be 5-dimensional (batches, n, n, n, channels)",
										   input_tensor.shape().DebugString()));

		// The last dimension for input is in_depth. It must be the same as the
		// filter's in_depth.
		const int64 in_depth = input_tensor.dim_size(4);

		const Tensor& node_num_data_tensor = context->input(1);
		auto node_num_data = node_num_data_tensor.flat<int64>();

		const Tensor& current_depth_tensor = context->input(2);
		auto current_depth = current_depth_tensor.flat<int64>()(0);

		// Create mapping for the key indices
		int n = 1 << current_depth;
		std::vector<unsigned> mapper(n*n*n);
		for (unsigned x = 0; x < n; ++x )
		{
			for (unsigned y = 0; y < n; ++y)
			{
				for (unsigned z = 0; z < n; ++z)
				{
					// xyz index
					unsigned xyz = (n*x + y) * n + z;

					// key
					unsigned key = 0;
					for (int i = 0; i < current_depth; i++)
					{
						unsigned mask = 1u << i;
						key |= ((x & mask) << (2 * i + 2)) |
							((y & mask) << (2 * i + 1)) |
							((z & mask) << (2 * i));
					}

					// mapping
					mapper[xyz] = key;
				}
			}
		}


		int voxel_num = 1 << 3 * current_depth;
		int channel = in_depth;
		int bottom_h = node_num_data(current_depth) * 3;

		//Create the full voxel layer
		TensorShape output_shape;
		output_shape.AddDim(input_tensor.dim_size(0));
		output_shape.AddDim(bottom_h);
		output_shape.AddDim(in_depth);

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
		auto output = output_tensor->flat<float>();

		auto xyz_to_key = mapper;

		const int batch_size = input_tensor.dim_size(0);
		for (int n = 0; n < batch_size; ++n) {
			for (int c = 0; c < channel; ++c) {
				for (int k = 0; k < voxel_num; ++k) {
					// This is literally the same as the forward version
					// but reversed
					output(c*bottom_h + n*voxel_num + xyz_to_key[k]) =
						input((n*channel + c)*voxel_num + k);

				}
			}
		}
	}
};

REGISTER_KERNEL_BUILDER(
    Name("OctreeFullLayerGradient")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    OctreeFullLayerGradientOp<float>);
