#include "musa_executor.hpp"

extern "C"
{
#include "graph/tensor.h"
#include "utility/log.h"
}


MUSAEngine::MUSAEngine()
{

}

void MUSAEngine::MUSADataMalloc(struct graph* ir_graph, int ir_tensor_idx)
{
    auto iter = this->gpu_addr_map.find(ir_tensor_idx);
    if (this->gpu_addr_map.end() == iter)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        void* gpu_data = nullptr;
        if(musaSuccess == musaMalloc((void **)&gpu_data, ir_tensor->elem_num * ir_tensor->elem_size))
        {
            TLOG_INFO(" musa malloc tensor(%d) name %s size %d addr %p\n",
                    ir_tensor->index, ir_tensor->name, ir_tensor->elem_num * ir_tensor->elem_size, gpu_data);
        }
        if ( TENSOR_TYPE_CONST == ir_tensor->tensor_type || TENSOR_TYPE_DEP == ir_tensor->tensor_type )
        {
            TLOG_INFO(" musa copy tensor(%d) name %s addr %p\n",ir_tensor->index, ir_tensor->name, gpu_data);
            musaMemcpy(gpu_data, ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size, musaMemcpyHostToDevice);
        }
        this->gpu_addr_map[ir_tensor_idx] = gpu_data;
    }
}

void MUSAEngine::DataUpload(struct graph* ir_graph, int ir_tensor_idx)
{
    struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
    musaMemcpy(this->gpu_addr_map[ir_tensor_idx], ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size, musaMemcpyHostToDevice);
}

void MUSAEngine::DataDownload(struct graph* ir_graph, int ir_tensor_idx)
{
    struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
    musaMemcpy(ir_tensor->data, this->gpu_addr_map[ir_tensor_idx], ir_tensor->elem_num * ir_tensor->elem_size, musaMemcpyDeviceToHost);
}

int MUSAEngine::Build(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;

        switch (op_type)
        {
            case OP_CLIP:
                this->AddClipNode(ir_graph, ir_node);
                break;
            case OP_CONCAT:
                this->AddConcatNode(ir_graph, ir_node);
                break;
            case OP_CONST:
                break;
            case OP_CONV:
                this->AddConvolutionNode(ir_graph, ir_node);
                break;
            case OP_DROPOUT:
                this->AddDropoutNode(ir_graph, ir_node);
                break;
            case OP_ELTWISE:
                this->AddEltwiseNode(ir_graph, ir_node);
                break;
            case OP_INPUT:
                break;
            case OP_FC:
                this->AddFullyConnectionNode(ir_graph, ir_node);
                break;
            case OP_FLATTEN:
                this->AddFlattenNode(ir_graph, ir_node);
                break;
            case OP_PERMUTE:
                this->AddPermuteNode(ir_graph, ir_node);
                break;
            case OP_POOL:
                this->AddPoolingNode(ir_graph, ir_node);
                break;
            case OP_RELU:
                this->AddReluNode(ir_graph, ir_node);
                break;
            case OP_RESHAPE:
                this->AddReshapeNode(ir_graph, ir_node);
                break;
            case OP_SLICE:
                this->AddSliceNode(ir_graph, ir_node);
                break;
            case OP_SOFTMAX:
                this->AddSoftmaxNode(ir_graph, ir_node);
            default:
                TLOG_INFO("Tengine GPU: Cannot support OP(%d).\n", ir_node->index);
                break;
        }
    }
    return 0;
}

int MUSAEngine::MUSAEnginePreRun(struct subgraph* subgraph)
{
    const auto musa_status = musaSetDevice(DEFAULT_DEVICE_ID);
    if (musa_status != musaSuccess)
    {
        fprintf(stderr, "Tengine: Cannot lock to socket %d.\n", DEFAULT_DEVICE_ID);
        return -1;
    }

    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (int j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            this->MUSADataMalloc(ir_graph, ir_tensor_idx);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            this->MUSADataMalloc(ir_graph, ir_tensor_idx);
        }
    }

    int val;
    musaDeviceGetAttribute(&val, musaDevAttrMaxThreadsPerBlock, 0);
    TLOG_INFO("musaDevAttrMaxThreadsPerBlock %d\n",val);
    this->Build(subgraph);

    mublasCreate(&this->mublas_handle);

    for (int i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        struct tensor* graph_out_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        graph_out_tensor->data = (void*)malloc(graph_out_tensor->elem_num * graph_out_tensor->elem_size);
    }

    return 0;
};

int MUSAEngine::MUSAEngineRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        this->DataUpload(ir_graph, ir_tensor_idx);
    }

    for (auto& func : this->ops)
    {
        func();
    }

    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        this->DataDownload(ir_graph, ir_tensor_idx);
    }

#ifdef DEBUG_DATA
    for (auto iter = this->gpu_addr_map.begin(); iter != this->gpu_addr_map.end(); iter++)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, iter->first);
        musaMemcpy(ir_tensor->data, iter->second, ir_tensor->elem_num * ir_tensor->elem_size, musaMemcpyDeviceToHost);
    }
#endif

    return 0;
}

void MUSAEngine::MUSAEnginePostRun()
{
    for (auto iter = this->gpu_addr_map.begin(); iter != this->gpu_addr_map.end(); iter++)
    {
        musaFree(iter->second);
    }
};
