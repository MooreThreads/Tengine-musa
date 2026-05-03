#include "musa_executor.hpp"
extern "C" {
#include "fc_param.h"
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

void fc_gpu_kernel(mublasHandle_t& handle, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    const float* d_A = (const float*)gpu_addr_map[input_tensor->index];
    const float* d_B = (const float*)gpu_addr_map[weight_tensor->index];
    float* d_C = (float*)gpu_addr_map[output_tensor->index];

    int hidden = input_tensor->dims[1];
    if (input_tensor->dim_num > 2) hidden = hidden * input_tensor->dims[2];
    if (input_tensor->dim_num > 3) hidden = hidden * input_tensor->dims[3];

    struct fc_param* fc_param = (struct fc_param*)ir_node->op.param_mem;

    int A_ROW = input_tensor->dims[0];
    int A_COL = hidden;
    int B_ROW = hidden;
    int B_COL = fc_param->num_output;

    int need_trans = 0;
    int weight_out = weight_tensor->dims[0];
    if (weight_out == fc_param->num_output) need_trans = 0;
    else need_trans = 1;

    float a = 1, b = 0;
    if (1 == need_trans)
    {
        mublasSgemm(handle, MUBLAS_OP_N, MUBLAS_OP_N,
            B_COL, A_ROW, B_ROW, &a, d_B, B_COL, d_A, A_COL, &b, d_C, B_COL);
    }
    else
    {
        mublasSgemm(handle, MUBLAS_OP_T, MUBLAS_OP_N,
            B_COL, A_ROW, B_ROW, &a, d_B, B_ROW, d_A, A_COL, &b, d_C, B_COL);
    }
}

void MUSAEngine::AddFullyConnectionNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_FC.\n", ir_node->index);
    mublasCreate(&this->mublas_handle);
    fc_gpu_kernel(this->mublas_handle, ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&fc_gpu_kernel, this->mublas_handle, ir_graph, ir_node, this->gpu_addr_map));
}
