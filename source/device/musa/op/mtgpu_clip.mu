#include "musa_executor.hpp"

extern "C"
{
#include "clip_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

#include <musa_runtime.h>

__global__ void relu6(float *y, float *x, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        y[idx] = x[idx] > 6 ? 6 : x[idx];
        y[idx] = y[idx] < 0 ? 0 : y[idx];
    }
}

void relu6_gpu_kernel(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int bs = 1024;
    int s = ceil((output_tensor->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);

    relu6<<<grid, bs>>>((float*)gpu_addr_map[output_tensor->index], (float*)gpu_addr_map[input_tensor->index], output_tensor->elem_num);
}

void MUSAEngine::AddClipNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_CLIP.\n", ir_node->index);
    relu6_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&relu6_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
