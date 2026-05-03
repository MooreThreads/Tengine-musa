#include "musa_executor.hpp"
extern "C" {
#include "permute_param.h"
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void permute(float *y, float *x, int elem_num, int n, int c, int hw)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int chw = c * hw;
    int idx_n = idx / chw;
    int idx_c = idx % chw / hw;
    int idx_hw = idx % hw;
    int idx_new = idx_n * chw + idx_hw * c + idx_c;
    if (idx < elem_num) { y[idx_new] = x[idx]; }
}

void permute_gpu_kernel(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct permute_param* param = (struct permute_param*)ir_node->op.param_mem;
    int bs = 1024; int s = ceil((output_tensor->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);
    if (param->order0 == 0 && param->order1 == 2 && param->order2 == 3 && param->order3 == 1)
        permute<<<grid, bs>>>((float*)gpu_addr_map[output_tensor->index], (float*)gpu_addr_map[input_tensor->index], output_tensor->elem_num,
                              input_tensor->dims[0], input_tensor->dims[1], input_tensor->dims[2] * input_tensor->dims[3]);
}

void MUSAEngine::AddPermuteNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_PERMUTE.\n", ir_node->index);
    permute_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&permute_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
