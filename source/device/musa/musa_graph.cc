#include "musa_graph.hpp"
#include "musa_executor.hpp"


int musa_dev_init(struct device* dev)
{
    (void)dev;
    return 0;
}


int musa_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    subgraph->device_graph = new MUSAEngine;
    auto engine = (MUSAEngine*)subgraph->device_graph;

    return engine->MUSAEnginePreRun(subgraph);
}


int musa_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (MUSAEngine*)subgraph->device_graph;
    return engine->MUSAEngineRun(subgraph);
}


int musa_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (MUSAEngine*)subgraph->device_graph;
    engine->MUSAEnginePostRun();
    delete engine;

    return 0;
}


int musa_dev_release(struct device* dev)
{
    (void)dev;
    return 0;
}
