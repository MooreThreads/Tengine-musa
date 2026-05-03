#pragma once

#include "device/device.h"

#define MUSA_DEV_NAME "MUSA"

extern "C" {
struct musa_device
{
    struct device base;
};

int register_musa_device(void);
}
