#pragma once

#include <map>
#include <vector>
#include <functional>
#include <cstdio>

extern "C" {
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
}

#include <mublas.h>
#include <mudnn.h>
#include <musa_runtime.h>

#define DEFAULT_DEVICE_ID 0

typedef std::map<uint32_t, uint32_t> dict_uint2uint;
typedef std::map<uint32_t, void*> dict_uint2voidx;
typedef std::function<void()> GPU_kernel;

class MUSAEngine
{
public:
    MUSAEngine();
    ~MUSAEngine() = default;

    int MUSAEnginePreRun(struct subgraph* subgraph);
    int MUSAEngineRun(struct subgraph* subgraph);
    void MUSAEnginePostRun();

private:
    void AddClipNode(struct graph* ir_graph, struct node* ir_node);
    void AddConcatNode(struct graph* ir_graph, struct node* ir_node);
    void AddConvolutionNode(struct graph* ir_graph, struct node* ir_node);
    void AddDropoutNode(struct graph* ir_graph, struct node* ir_node);
    void AddEltwiseNode(struct graph* ir_graph, struct node* ir_node);
    void AddFullyConnectionNode(struct graph* ir_graph, struct node* ir_node);
    void AddFlattenNode(struct graph* ir_graph, struct node* ir_node);
    void AddPermuteNode(struct graph* ir_graph, struct node* ir_node);
    void AddPoolingNode(struct graph* ir_graph, struct node* ir_node);
    void AddReluNode(struct graph* ir_graph, struct node* ir_node);
    void AddReshapeNode(struct graph* ir_graph, struct node* ir_node);
    void AddSliceNode(struct graph* ir_graph, struct node* ir_node);
    void AddSoftmaxNode(struct graph* ir_graph, struct node* ir_node);

private:
    void MUSADataMalloc(struct graph* ir_graph, int ir_tensor_idx);
    int Build(struct subgraph* subgraph);
    void DataUpload(struct graph* ir_graph, int ir_tensor_idx);
    void DataDownload(struct graph* ir_graph, int ir_tensor_idx);

private:
    std::vector<GPU_kernel> ops;

private:
    musa::dnn::Handle mudnn_handle;
    mublasHandle_t mublas_handle;

public:
    dict_uint2voidx gpu_addr_map;
};
