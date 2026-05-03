#include "musa_executor.hpp"

extern "C"
{
#include "convolution_param.h"
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void bias_add(float *y, float *x, int elem_num_perimg, int elem_perchannel, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) { y[idx] += x[idx % elem_num_perimg / elem_perchannel]; }
}

__global__ void bias_add_relu(float *y, float *x, int elem_num_perimg, int elem_perchannel, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        y[idx] += x[idx % elem_num_perimg / elem_perchannel];
        y[idx] = y[idx] > 0 ? y[idx] : 0;
    }
}

void conv_gpu_kernel(musa::dnn::Handle* handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map, int setalgo)
{
    struct tensor* conv_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* conv_weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* conv_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

    musa::dnn::Convolution conv_op;
    int pad[2] = {conv_param->pad_h0, conv_param->pad_w0};
    int stride[2] = {conv_param->stride_h, conv_param->stride_w};
    int dilation[2] = {conv_param->dilation_h, conv_param->dilation_w};
    conv_op.SetNdInfo(2, pad, stride, dilation);
    conv_op.SetGroups(conv_param->group);

    musa::dnn::Tensor input_t, weight_t, output_t;
    input_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    input_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t in_dims[4] = {conv_input_data->dims[0], conv_input_data->dims[1], conv_input_data->dims[2], conv_input_data->dims[3]};
    input_t.SetNdInfo(4, in_dims);
    input_t.SetAddr(gpu_addr_map[conv_input_data->index]);

    weight_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    weight_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t w_dims[4] = {conv_weight->dims[0], conv_weight->dims[1], conv_weight->dims[2], conv_weight->dims[3]};
    weight_t.SetNdInfo(4, w_dims);
    weight_t.SetAddr(gpu_addr_map[conv_weight->index]);

    output_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    output_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t out_dims[4] = {conv_output_data->dims[0], conv_output_data->dims[1], conv_output_data->dims[2], conv_output_data->dims[3]};
    output_t.SetNdInfo(4, out_dims);
    output_t.SetAddr(gpu_addr_map[conv_output_data->index]);

    musa::dnn::Convolution::Algorithm algo;
    conv_op.GetRecommendForwardAlgorithm(*handle, algo, output_t, input_t, weight_t);

    size_t ws_size = 0;
    conv_op.GetForwardWorkspaceSize(*handle, ws_size, output_t, input_t, weight_t, algo);
    void* workspace = nullptr;
    if (ws_size > 0) musaMalloc(&workspace, ws_size);

    musa::dnn::MemoryMaintainer mem_handler = [](size_t) -> musa::dnn::MemoryHandler { return {nullptr, [](void*){}}; };
    if (ws_size > 0) {
        void* ws_ptr = workspace;
        mem_handler = [ws_ptr](size_t) -> musa::dnn::MemoryHandler {
            return {ws_ptr, [](void*){}};
        };
    }
    conv_op.Run(*handle, output_t, input_t, weight_t, algo, mem_handler);

    int bs = 1024;
    int s = ceil((conv_output_data->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);

    if (2 < ir_node->input_num)
    {
        struct tensor* conv_bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int elem_23 = conv_output_data->dims[2] * conv_output_data->dims[3];
        int elem_123 = conv_output_data->dims[1] * elem_23;

        if (conv_param->activation == 0)
            bias_add_relu<<<grid, bs>>>((float*)gpu_addr_map[conv_output_data->index], (float*)gpu_addr_map[conv_bias->index], elem_123, elem_23, conv_output_data->elem_num);
        else
            bias_add<<<grid, bs>>>((float*)gpu_addr_map[conv_output_data->index], (float*)gpu_addr_map[conv_bias->index], elem_123, elem_23, conv_output_data->elem_num);
    }

    if (workspace) musaFree(workspace);
}

void MUSAEngine::AddConvolutionNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_CONV.\n", ir_node->index);
    conv_gpu_kernel(&this->mudnn_handle, ir_graph, ir_node, this->gpu_addr_map, 0);
    this->ops.push_back(std::bind(&conv_gpu_kernel, &this->mudnn_handle, ir_graph, ir_node, this->gpu_addr_map, 1));
}
