#include "musa_executor.hpp"
extern "C" {
#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void reshape(float *y, float *x, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) { y[idx] = x[idx]; }
}

void reshape_gpu_kernel(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx gpu_addr_map)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int bs = 1024; int s = ceil((output_tensor->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);
    reshape<<<grid, bs>>>((float*)gpu_addr_map[output_tensor->index], (float*)gpu_addr_map[input_tensor->index], output_tensor->elem_num);
}

void MUSAEngine::AddReshapeNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_RESHAPE.\n", ir_node->index);
    reshape_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&reshape_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
