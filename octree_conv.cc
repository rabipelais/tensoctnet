#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("OctreeConv")
    .Input("input: T")
    .Input("filter: T")
    .Input("final_nodes: int32")
    .Input("key_data: int32")
    .Input("children_data: int32")
    .Input("current_depth: int32")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true");

template <typename Device, typename T>
class OctreeConvOp : public BinaryOp<T> {
 public:
  explicit OctreeConvOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();

    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    const int64 stride_h = GetTensorDim(strides_, data_format_, 'H');
    const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride and dilation from the second and third
    // dimensions only (we do not support striding or dilation on the batch or
    // depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_rows, filter_rows, dilation_rows,
                                stride_rows, padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_cols, filter_cols, dilation_cols,
                                stride_cols, padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv2D: in_depth = " << in_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", dilation_rows = " << dilation_rows
            << ", dilation_cols = " << dilation_cols
            << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

#ifdef TENSORFLOW_USE_LIBXSMM
    if (LaunchXsmmConvOp<Device, T>::Run(
            context, input, filter, batch, input_rows, input_cols, in_depth,
            filter_rows, filter_cols, pad_rows, pad_cols, out_rows, out_cols,
            out_depth, dilation_rows, dilation_cols, stride_rows, stride_cols,
            output, data_format_)) {
      return;
    }
#endif

    if (LaunchDeepConvOp<Device, T>::Run(
            context, input, filter, batch, input_rows, input_cols, in_depth,
            filter_rows, filter_cols, pad_rows, pad_cols, out_rows, out_cols,
            out_depth, dilation_rows, dilation_cols, stride_rows, stride_cols,
            output, data_format_)) {
      return;
    }

    launcher_(context, use_cudnn_, cudnn_use_autotune_, input, filter,
              dilation_rows, dilation_cols, stride_rows, stride_cols, padding_,
              output, data_format_);
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  LaunchConv2DOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DOp);
};

void init_neigh_index()
{
	// ni for kernel_size=3
	vector<int> shape{ 216 };
	ni_[3].reset(new Blob<int>(shape));
	int* ni3 = ni_[3]->mutable_cpu_data();
	int id = 0;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			for (int k = 0; k < 2; ++k)
			{
				for (int x = 0; x < 3; ++x)
				{
					for (int y = 0; y < 3; ++y)
					{
						for (int z = 0; z < 3; ++z)
						{
							ni3[id++] = (x + i << 4) | (y + j << 2) | z + k;
						}
					}
				}
			}
		}
	}

	// ni for kernel_size=2
	shape[0] = 64;
	ni_[2].reset(new Blob<int>(shape));
	int* ni2 = ni_[2]->mutable_cpu_data();
	const int arr[] = { 13, 14, 16, 17, 22, 23, 25, 26 };
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			ni2[i * 8 + j] = ni3[i * 27 + arr[j]];
		}
	}

	// ni for kernel_size=1
	shape[0] = 8;
	ni_[1].reset(new Blob<int>(shape));
	int* ni1 = ni_[1]->mutable_cpu_data();
	for (int i = 0; i < 8; ++i)
	{
		ni1[i] = ni3[i * 27 + 13];
	}


	// init the array parent & displacement
	id = 0;
	int tmp[64];
	shape[0] = 64;
	displacement_.Reshape(shape);
	int* dis_ptr = displacement_.mutable_cpu_data();
	for (int x = 1; x < 5; ++x)
	{
		for (int y = 1; y < 5; ++y)
		{
			for (int z = 1; z < 5; ++z)
			{
				int x1 = x / 2;
				int xb = x % 2;
				int y1 = y / 2;
				int yb = y % 2;
				int z1 = z / 2;
				int zb = z % 2;

				tmp[id] = x1 * 9 + y1 * 3 + z1;
				dis_ptr[id] = (xb << 2) | (yb << 1) | zb;
				id++;
			}
		}
	}

	shape[0] = 512;
	parent_.Reshape(shape);
	int* parent_ptr = parent_.mutable_cpu_data();
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 64; ++j)
		{
			parent_ptr[i * 64 + j] = ni3[i * 27 + tmp[j]];
		}
	}
}
