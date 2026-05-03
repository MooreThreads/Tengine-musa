#include "musa_executor.hpp"
extern "C" {
#include "slice_param.h"
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void slice(float *y, float *x, int elem_num, int res)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_new = idx + res;
    if (idx < elem_num) { y[idx] = x[idx_new]; }
}

void slice_gpu_kernel(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int bs = 1024; int s = ceil((output_tensor->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);
    struct slice_param* param = (struct slice_param*)ir_node->op.param_mem;
    int res = 1;
    for (uint8_t i = input_tensor->dim_num-1; i > param->axis; i--)
        res *= input_tensor->dims[i];
    res *= param->begin;
    slice<<<grid, bs>>>((float*)gpu_addr_map[output_tensor->index], (float*)gpu_addr_map[input_tensor->index], output_tensor->elem_num, res);
}

void MUSAEngine::AddSliceNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_SLICE.\n", ir_node->index);
    slice_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&slice_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
