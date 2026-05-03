#include "musa_executor.hpp"

extern "C"
{
#include "softmax_param.h"
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void softmax_max(float* k, float* result_block)
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = k[i];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) result_block[blockIdx.x] = sdata[0];
}

__global__ void softmax_k_upload(float* k, float* x, int N, int elem_perchannel_match, int elem_perchannel)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int idx_mul = i / elem_perchannel_match;
        int idx_new = i % elem_perchannel_match;
        if (elem_perchannel > idx_new) k[i] = x[idx_mul * elem_perchannel + idx_new];
        else k[i] = -9999.9f;
    }
}

__global__ void softmax_k_download(float* k, float* x, int N, int elem_perchannel_match, int elem_perchannel)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx_mul = idx / elem_perchannel_match;
        int idx_new = idx % elem_perchannel_match;
        if (elem_perchannel > idx_new) x[idx_mul * elem_perchannel + idx_new] = k[idx];
    }
}

__global__ void softmax_exp_sum(float* k, float* result_block, int elem_perchannel_match, int elem_perchannel)
{
    __shared__ float sdata[1024];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_perchannel <= (i % elem_perchannel_match)) k[i] = 0;
    else k[i] = exp(k[i] - result_block[blockIdx.x]);
    __syncthreads();
    int tid = threadIdx.x;
    if (elem_perchannel <= (i % elem_perchannel_match)) sdata[tid] = 0;
    else sdata[tid] = k[i];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result_block[blockIdx.x] = sdata[0];
    __syncthreads();
}

__global__ void softmax_exp_div(float* k, float* result_block, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) k[i] = float(k[i] / result_block[blockIdx.x]);
}

void softmax_gpu_kernel_mudnn(musa::dnn::Handle* handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map, int axis)
{
    struct tensor* soft_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* soft_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    musa::dnn::Softmax softmax_op;
    softmax_op.SetDim(axis);
    softmax_op.SetAlgorithm(musa::dnn::Softmax::Algorithm::ACCURATE);
    softmax_op.SetMode(musa::dnn::Softmax::Mode::SOFTMAX);

    musa::dnn::Tensor input_t, output_t;
    input_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    input_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t in_dims[4] = {soft_input_data->dims[0], soft_input_data->dims[1], soft_input_data->dims[2], soft_input_data->dims[3]};
    input_t.SetNdInfo(4, in_dims);
    input_t.SetAddr(gpu_addr_map[soft_input_data->index]);

    output_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    output_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t out_dims[4] = {soft_output_data->dims[0], soft_output_data->dims[1], soft_output_data->dims[2], soft_output_data->dims[3]};
    output_t.SetNdInfo(4, out_dims);
    output_t.SetAddr(gpu_addr_map[soft_output_data->index]);

    softmax_op.Run(*handle, output_t, input_t);
}

void softmax_gpu_kernel_custom(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct softmax_param* param = (struct softmax_param*)ir_node->op.param_mem;

    int channels = 1;
    for (int i = 0; i < param->axis; i++) channels *= output_tensor->dims[i];

    int elem_perchannel = output_tensor->elem_num / channels;
    int bs = (elem_perchannel / 32 + 1) * 32;
    if (bs > 1024) bs = 1024;
    int elem_perchannel_match = ((elem_perchannel - 1) / bs + 1) * bs;
    int s = elem_perchannel_match * channels / bs;
    dim3 grid = dim3(s);

    float* k = NULL;
    musaMalloc((void**)&k, elem_perchannel_match * channels * sizeof(float));
    softmax_k_upload<<<grid, bs>>>(k, (float*)gpu_addr_map[input_tensor->index], elem_perchannel_match * channels, elem_perchannel_match, elem_perchannel);

    float* result_block = NULL;
    musaMalloc((void**)&result_block, s * sizeof(float));
    softmax_max<<<grid, bs>>>(k, result_block);

    if (s != channels)
    {
        float* result = (float*)malloc(channels * sizeof(float));
        float* result_tmp = (float*)malloc(s * sizeof(float));
        musaMemcpy(result_tmp, result_block, s * sizeof(float), musaMemcpyDeviceToHost);
        for (int i = 0; i < channels; i++) {
            result[i] = -9999.9f;
            for (int j = 0; j < s / channels; j++)
                result[i] = max(result[i], result_tmp[i * s / channels + j]);
            for (int j = 0; j < s / channels; j++)
                result_tmp[i * s / channels + j] = result[i];
        }
        musaMemcpy(result_block, result_tmp, s * sizeof(float), musaMemcpyHostToDevice);
        free(result);
        free(result_tmp);
    }

    softmax_exp_sum<<<grid, bs>>>(k, result_block, elem_perchannel_match, elem_perchannel);
    softmax_exp_div<<<grid, bs>>>(k, result_block, elem_perchannel_match * channels);
    softmax_k_download<<<grid, bs>>>(k, (float*)gpu_addr_map[output_tensor->index], elem_perchannel_match * channels, elem_perchannel_match, elem_perchannel);

    musaFree(result_block);
    musaFree(k);
}

void MUSAEngine::AddSoftmaxNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_SOFTMAX.\n", ir_node->index);
    struct softmax_param* param = (struct softmax_param*)ir_node->op.param_mem;
    switch (param->axis)
    {
        case 0:
        case 1:
            softmax_gpu_kernel_mudnn(&this->mudnn_handle, ir_graph, ir_node, this->gpu_addr_map, param->axis);
            this->ops.push_back(std::bind(&softmax_gpu_kernel_mudnn, &this->mudnn_handle, ir_graph, ir_node, this->gpu_addr_map, param->axis));
            break;
        case 2:
            softmax_gpu_kernel_custom(ir_graph, ir_node, this->gpu_addr_map);
            this->ops.push_back(std::bind(&softmax_gpu_kernel_custom, ir_graph, ir_node, this->gpu_addr_map));
            break;
        default:
            TLOG_INFO("Tengine GPU: Cannot support SOFTMAX axis(%d).\n", param->axis);
            break;
    }
}
