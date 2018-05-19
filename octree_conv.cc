#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <unordered_map>

using namespace tensorflow;

void calc_neigh_cpu(std::vector<int> &neigh, const int depth, const int batch_size);

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
		int ni3[216];
		int id = 0;
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 2; ++j)
				for (int k = 0; k < 2; ++k)
					for (int x = 0; x < 3; ++x)
						for (int y = 0; y < 3; ++y)
							for (int z = 0; z < 3; ++z)
								ni3[id++] = (x + i << 4) | (y + j << 2) | z + k;

		// Voxels neighbours
		std::vector<int> neigh((1 << 3 * current_depth) * 8);
		calc_neigh_cpu(neigh, current_depth, 1);

		// octree2col
		const int octree_h = current_depth << 3 * (stride - 1); // = * 8
		const int kernel_size = kernel_tensor.dim_size(0) * kernel_tensor.dim_size(1) * kernel_tensor.dim_size(2); // only tested for 3 * 3 * 3, should probably check for this


		std::vector<float> data_col(in_depth * kernel_size * out_size);

		for(int c = 0; c < in_depth; ++c) {
			for(int k = 0; k < kernel_size; ++k) {
				for(int h = 0; h < out_size; ++h) {

					const int index = stride == 2 ? (h << 6) + ni3[k] :
						(h >> 3 << 6) + ni3[(h % 8) * kernel_size + k];

					const int p = neigh[index];

					data_col[(c*kernel_size + k)*out_size + h] = p == -1 ?
						0 : input(c*octree_h + p);
				}
			}
		}

		auto data_col_eigen = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(data_col.data(), out_size, in_depth * kernel_size);

		auto kernel_flat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(kernel_tensor.flat<float>().data(), in_depth * kernel_size, out_depth);

		auto out_eigen = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(output.data(), out_size, out_depth);

		out_eigen.noalias() = data_col_eigen * kernel_flat;
	}
};

void calc_neigh_cpu(std::vector<int> &neigh, const int depth, const int batch_size)
{
	unsigned node_num = 1 << 3 * depth;
	const unsigned  bound = 1 << depth;
	for (unsigned n = 0; n < batch_size; ++n)
	{
		for (unsigned i = 0; i < node_num; i += 8)
		{
			// key to xyz
			unsigned x0 = 0, y0 = 0, z0 = 0;
			for (unsigned d = 0; d < depth; d++)
			{
				x0 |= (i & (1 << 3 * d + 2)) >> (2 * d + 2);
				y0 |= (i & (1 << 3 * d + 1)) >> (2 * d + 1);
				z0 |= (i & (1 << 3 * d + 0)) >> (2 * d + 0);
			}

			for (unsigned x = 0; x < 4; ++x)
			{
				unsigned x1 = x0 + x - 1;
				if (x1 & bound) continue;
				for (unsigned y = 0; y < 4; ++y)
				{
					unsigned y1 = y0 + y - 1;
					if (y1 & bound) continue;
					for (unsigned z = 0; z < 4; ++z)
					{
						int z1 = z0 + z - 1;
						if (z1 & bound) continue;

						// xyz index
						unsigned xyz = (x << 4) | (y << 2) | z;

						// key
						unsigned key1 = 0;
						for (int d = 0; d < depth; d++)
						{
							unsigned mask = 1u << d;
							key1 |= ((x1 & mask) << (2 * d + 2)) |
								((y1 & mask) << (2 * d + 1)) |
								((z1 & mask) << (2 * d));
						}

						// mapping
						neigh[xyz + i * 8 + n*node_num * 8] = key1 + n*node_num;
					}
				}
			}
		}
	}
}


// // Fast version with hash map
// void calc_neighbor(int* neigh, const unsigned* key, const int node_num)
// 	{
// 		typedef unsigned char ubyte;

// 		// build hash table
// 		vector<std::pair<unsigned, int> > entries(node_num);
// 		for (int id = 0; id < node_num; ++id)
// 		{	// ignore the root node
// 			entries[id] = std::make_pair(key[id], id + displacement);
// 		}
// 		std::unordered_map<unsigned, int> hash_table(entries.begin(), entries.end());

// 		// calc neighborhood
// 		for (int id = 0; id < node_num; id += 8)
// 		{
// 			// the neighborhood volume
// 			int* ngh = neigh + id * 8;
// 			const ubyte* k0 = (const ubyte*)(key + id);
// 			// currently the maximize octree depth is 8
// 			ubyte k1[4] = { 0, 0, 0, k0[3] };
// 			const ubyte bound = (1 << k0[3]) - 2;
// 			for (ubyte x = 0; x < 4; ++x)
// 			{
// 				k1[0] = k0[0] + x - 1;
// 				for (ubyte y = 0; y < 4; ++y)
// 				{
// 					k1[1] = k0[1] + y - 1;
// 					for (ubyte z = 0; z < 4; ++z)
// 					{
// 						k1[2] = k0[2] + z - 1;

// 						// find
// 						unsigned* k2 = reinterpret_cast<unsigned*>(k1);
// 						auto rst = hash_table.find(*k2);
// 						ubyte i = (x << 4) | (y << 2) | z;
// 						if (rst != hash_table.end())
// 						{
// 							ngh[i] = rst->second;
// 						}
// 						else {
// 							ngh[i] = -1;
// 						}
// 					}
// 				}
// 			}
// 		}
// }


REGISTER_KERNEL_BUILDER(
    Name("OctreeConv")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    OctreeConvOp<float>);
