#pragma once

extern "C" {
#include "operator/op.h"
}

const int musa_supported_ops[] = {
    OP_CLIP,
    OP_CONCAT,
    OP_CONST,
    OP_CONV,
    OP_DROPOUT,
    OP_ELTWISE,
    OP_FC,
    OP_FLATTEN,
    OP_INPUT,
    OP_PERMUTE,
    OP_POOL,
    OP_RELU,
    OP_RESHAPE,
    OP_SLICE,
    OP_SOFTMAX};
