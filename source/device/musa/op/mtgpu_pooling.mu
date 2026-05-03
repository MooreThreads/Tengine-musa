#include "musa_executor.hpp"

extern "C"
{
#include "pooling_param.h"
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

void pooling_gpu_kernel(musa::dnn::Handle* handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map)
{
    struct tensor* pool_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* pool_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct pool_param* pool_param = (struct pool_param*)ir_node->op.param_mem;

    musa::dnn::Pooling pool_op;
    switch (pool_param->pool_method)
    {
        case 0:
            pool_op.SetMode(musa::dnn::Pooling::Mode::MAXPOOL);
            break;
        case 1:
            pool_op.SetMode(musa::dnn::Pooling::Mode::AVGPOOL_COUNT_PAD);
            break;
        default:
            fprintf(stderr, "don't support this method pooling\n");
            return;
    }

    int kernel[2] = {pool_param->kernel_h, pool_param->kernel_w};
    int pad[2] = {pool_param->pad_h0, pool_param->pad_w0};
    int stride[2] = {pool_param->stride_h, pool_param->stride_w};
    int dilation[2] = {1, 1};
    pool_op.SetNdInfo(2, kernel, pad, stride, dilation);

    musa::dnn::Tensor input_t, output_t, indices_t;
    input_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    input_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t in_dims[4] = {pool_input_data->dims[0], pool_input_data->dims[1], pool_input_data->dims[2], pool_input_data->dims[3]};
    input_t.SetNdInfo(4, in_dims);
    input_t.SetAddr(gpu_addr_map[pool_input_data->index]);

    output_t.SetType(musa::dnn::Tensor::Type::FLOAT);
    output_t.SetFormat(musa::dnn::Tensor::Format::NCHW);
    int64_t out_dims[4] = {pool_output_data->dims[0], pool_output_data->dims[1], pool_output_data->dims[2], pool_output_data->dims[3]};
    output_t.SetNdInfo(4, out_dims);
    output_t.SetAddr(gpu_addr_map[pool_output_data->index]);

    auto status = pool_op.Run(*handle, output_t, input_t, indices_t);
    if (status != musa::dnn::Status::SUCCESS)
    {
        fprintf(stderr, "GPU: Fail to forward pooling!\n");
    }
}

void MUSAEngine::AddPoolingNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_POOL.\n", ir_node->index);
    pooling_gpu_kernel(&this->mudnn_handle, ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&pooling_gpu_kernel, &this->mudnn_handle, ir_graph, ir_node, this->gpu_addr_map));
}
