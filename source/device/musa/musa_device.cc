#include "musa_device.hpp"

#include "musa_limit.hpp"
#include "musa_graph.hpp"

extern "C"
{
#include "api/c_api.h"
#include "executer/executer.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "optimizer/split.h"
#include "utility/vector.h"
#include "utility/log.h"
}

#include "cstring"


int musa_describe(struct device* device, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    (void)device;

    for (int op_type : musa_supported_ops)
    {
        push_vector_data(allowed_ops, &op_type);
    }

    for (int i = 0; i < OP_BUILTIN_LAST; i++)
    {
        bool in_list = false;

        for (const auto& type : musa_supported_ops)
        {
            if (type == i)
            {
                in_list = true;
                break;
            }
        }

        if (!in_list)
        {
            push_vector_data(blocked_ops, &i);
        }
    }

    int precision_var = TENGINE_DT_UINT8;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP16;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP32;
    push_vector_data(precision, &precision_var);

    return 0;
}

int musa_evaluation(struct device* device, struct subgraph* sub_graph, struct vector* evolution_tensors, struct vector* evolution_nodes)
{
    (void)sub_graph;
    (void)evolution_tensors;
    (void)evolution_nodes;

    return 0;
}

int musa_allocate(struct device* device, struct subgraph* sub_graph)
{
    if (nullptr == device)
    {
        return -1;
    }

    sub_graph->input_wait_count = 0;

    for (int i = 0; i < sub_graph->input_num; i++)
    {
        struct tensor* tensor = get_ir_graph_tensor(sub_graph->graph, sub_graph->input_tensor_list[i]);

        if (tensor->tensor_type == TENSOR_TYPE_VAR)
            sub_graph->input_wait_count++;
    }

    return 0;
}

int musa_release(struct device* device, struct subgraph* sub_graph)
{
    (void)sub_graph;

    if (nullptr == device || !strcmp(MUSA_DEV_NAME, device->name))
    {
        return -1;
    }

    return 0;
}


int musa_split_graph(struct graph* ir_graph)
{
    struct device* cur_dev = ir_graph->attribute->context->device;

    if (0 != strcmp(MUSA_DEV_NAME, cur_dev->name))
    {
        return -1;
    }

    struct vector* allowed_ops = create_vector(sizeof(int), nullptr);
    struct vector* blocked_ops = create_vector(sizeof(int), nullptr);
    struct vector* precision = create_vector(sizeof(int), nullptr);

    cur_dev->allocator->describe(cur_dev, allowed_ops, blocked_ops, precision);

    split_graph_node_to_sub_graph(ir_graph, allowed_ops, blocked_ops, precision);

    release_vector(allowed_ops);
    release_vector(blocked_ops);
    release_vector(precision);

    generate_sub_graph_io(ir_graph);
    add_sub_graph_to_ir_graph(ir_graph);

    for (int i = 0; i < (uint16_t)get_vector_num(ir_graph->subgraph_list); i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->index = i;

        for (uint16_t j = 0; j < sub_graph->node_num; j++)
        {
            uint16_t node_id = sub_graph->node_list[j];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            ir_node->subgraph_idx = sub_graph->index;
        }
    }

    return 0;
}

extern "C"
{
static struct interface musa_interface = {
        .init           = musa_dev_init,
        .pre_run        = musa_dev_prerun,
        .run            = musa_dev_run,
        .post_run       = musa_dev_postrun,
        .async_run      = nullptr,
        .async_wait     = nullptr,
        .release_graph  = nullptr,
        .release_device = musa_dev_release,
};


static struct allocator musa_allocator = {
        .describe       = musa_describe,
        .evaluation     = musa_evaluation,
        .allocate       = musa_allocate,
        .release        = musa_release,
};


static struct optimizer musa_optimizer = {
        .split_graph    = musa_split_graph,
        .optimize_graph = nullptr,
};



static struct musa_device musa_device = {
        .base = {
                .name       = MUSA_DEV_NAME,
                .interface  = &musa_interface,
                .allocator  = &musa_allocator,
                .optimizer  = &musa_optimizer,
                .scheduler  = nullptr,
                .privacy    = nullptr,
        },
};


int register_musa_device(void)
{
    int ret = register_device(&musa_device.base);
    if (0 != ret)
    {
        TLOG_INFO("Tengine plugin %s register failed.\n", musa_device.base.name);
        return -1;
    }

    TLOG_INFO("Tengine plugin device %s is registered.\n", musa_device.base.name);
    return 0;
}


int unregister_musa_device(void)
{
    int ret = unregister_device(&musa_device.base);
    if (0 != ret)
    {
        TLOG_INFO("Tengine plugin %s unregister failed.\n", musa_device.base.name);
        return ret;
    }

    TLOG_INFO("Tengine plugin device %s is unregistered.\n", musa_device.base.name);

    return 0;
}
}
