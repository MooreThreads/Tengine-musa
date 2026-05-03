#pragma once

extern "C" {
#include "device/device.h"
#include "graph/subgraph.h"

int musa_dev_init(struct device* dev);
int musa_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options);
int musa_dev_run(struct device* dev, struct subgraph* subgraph);
int musa_dev_postrun(struct device* dev, struct subgraph* subgraph);
int musa_dev_release(struct device* dev);
}
