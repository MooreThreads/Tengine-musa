# Tengine MUSA 迁移报告

## 项目概述

**项目名称：** Tengine（tengine-lite）
**项目类型：** Lib（CMake 构建的 C/C++ 推理引擎库）
**原始仓库分支：** `official-Tengine`（基于 `tengine-lite`）
**迁移分支：** `musa-Tengine`

## 迁移范围

Tengine 是一个轻量级推理引擎，支持多种后端（CPU、CUDA、OpenCL、TensorRT、Vulkan 等）。CUDA 后端位于 `source/device/cuda/`，包含：

- **13 个 .cu kernel 文件**（算子实现）
- **4 个 .cc/.hpp 文件**（设备注册、图执行引擎）
- **1 个 CMakeLists.txt**（设备构建配置）
- **无 Python CUDA 依赖**

## API 映射

### CUDA Runtime API → MUSA Runtime API
| CUDA API | MUSA API |
|----------|----------|
| `cudaMalloc` | `musaMalloc` |
| `cudaFree` | `musaFree` |
| `cudaMemcpy` | `musaMemcpy` |
| `cudaSetDevice` | `musaSetDevice` |
| `cudaDeviceGetAttribute` | `musaDeviceGetAttribute` |
| `cudaSuccess` | `musaSuccess` |
| `cudaMemcpyHostToDevice` | `musaMemcpyHostToDevice` |
| `cudaMemcpyDeviceToHost` | `musaMemcpyDeviceToHost` |
| `cudaMemcpyDeviceToDevice` | `musaMemcpyDeviceToDevice` |

### cuBLAS → muBLAS
| cuBLAS API | muBLAS API |
|------------|------------|
| `cublasCreate` | `mublasCreate` |
| `cublasSgemm` | `mublasSgemm` |
| `cublasHandle_t` | `mublasHandle_t` |
| `CUBLAS_OP_N` | `MUBLAS_OP_N` |
| `CUBLAS_OP_T` | `MUBLAS_OP_T` |

### cuDNN → muDNN（API 重写）

muDNN 使用 C++ 类接口，与 cuDNN 的 C 函数式 API 完全不同。以下算子进行了完整的 API 重写：

| 算子 | cuDNN 实现方式 | muDNN 实现方式 |
|------|---------------|---------------|
| Convolution | `cudnnConvolutionForward()` + 描述符 | `musa::dnn::Convolution::Run()` + Tensor 类 |
| Pooling | `cudnnPoolingForward()` + 描述符 | `musa::dnn::Pooling::Run()` + Tensor 类 |
| Softmax (axis 0,1) | `cudnnSoftmaxForward()` + 描述符 | `musa::dnn::Softmax::Run()` + Tensor 类 |
| Softmax (axis 2) | 自定义 CUDA kernel | 自定义 MUSA kernel（直接移植） |

## 新增文件列表

| 文件路径 | 说明 |
|---------|------|
| `cmake/musa.cmake` | MUSA SDK 检测与配置 |
| `source/device/musa/CMakeLists.txt` | MUSA 设备构建配置 |
| `source/device/musa/musa_device.hpp` | MUSA 设备定义头文件 |
| `source/device/musa/musa_device.cc` | MUSA 设备注册与分图 |
| `source/device/musa/musa_executor.hpp` | MUSA 执行引擎头文件 |
| `source/device/musa/musa_executor.cc` | MUSA 执行引擎实现 |
| `source/device/musa/musa_graph.hpp` | MUSA 图执行接口 |
| `source/device/musa/musa_graph.cc` | MUSA 图执行实现 |
| `source/device/musa/musa_limit.hpp` | MUSA 支持算子列表 |
| `source/device/musa/op/mtgpu_clip.mu` | Clip/ReLU6 算子 |
| `source/device/musa/op/mtgpu_concat.mu` | Concat 算子 |
| `source/device/musa/op/mtgpu_convolution.mu` | Convolution 算子（muDNN） |
| `source/device/musa/op/mtgpu_dropout.mu` | Dropout 算子 |
| `source/device/musa/op/mtgpu_eltwise.mu` | Eltwise 算子 |
| `source/device/musa/op/mtgpu_fc.mu` | 全连接层（muBLAS） |
| `source/device/musa/op/mtgpu_flatten.mu` | Flatten 算子 |
| `source/device/musa/op/mtgpu_permute.mu` | Permute 算子 |
| `source/device/musa/op/mtgpu_pooling.mu` | Pooling 算子（muDNN） |
| `source/device/musa/op/mtgpu_relu.mu` | ReLU/LeakyReLU 算子 |
| `source/device/musa/op/mtgpu_reshape.mu` | Reshape 算子 |
| `source/device/musa/op/mtgpu_slice.mu` | Slice 算子 |
| `source/device/musa/op/mtgpu_softmax.mu` | Softmax 算子（muDNN + 自定义kernel） |

## 修改文件列表

| 文件路径 | 修改说明 |
|---------|---------|
| `CMakeLists.txt` | 添加 `TENGINE_ENABLE_MUSA` 选项和 `musa.cmake` 引入 |
| `source/CMakeLists.txt` | 添加 .mu 文件自定义编译规则（mcc 编译器） |
| `source/device/CMakeLists.txt` | 添加 MUSA 设备子目录和注册 |

## 构建与运行命令

### 在新环境中编译

```bash
# 前置条件：安装 MUSA SDK（mcc 编译器、muBLAS、muDNN 库）
# 确保 /usr/local/musa 目录存在

cd /workspace/repos/Tengine
git checkout musa-Tengine
mkdir build && cd build

# 仅启用 MUSA 后端
cmake .. -DTENGINE_ENABLE_MUSA=ON -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 安装
make install
```

### 运行测试

```bash
cd build
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

# 需要先准备测试模型文件（.tmfile）
# 测试使用 CPU 后端运行 ONNX 算子验证
cmake .. -DTENGINE_ENABLE_MUSA=ON -DTENGINE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## 测试结果

- **编译状态：** ✅ 100% 编译通过（无错误、无警告）
- **单元测试：** 75 个 ONNX 算子测试因缺少测试模型文件（`.tmfile`）而无法执行。这是 Tengine 测试框架的已知前提条件，与 MUSA 移植无关。库可正常加载并初始化 MUSA 设备。

## 支持的算子

MUSA 后端支持与原 CUDA 后端完全相同的 15 个算子：
Clip, Concat, Const, Conv, Dropout, Eltwise, FC, Flatten, Input, Permute, Pool, ReLU, Reshape, Slice, Softmax

## 已知限制

1. cuDNN 与 muDNN API 差异较大（C vs C++），Convolution/Pooling/Softmax 算子进行了完整重写
2. GPU 架构默认目标为 `mp_22`（S4000），可在 CMake 中调整
3. 测试需要额外的模型文件准备步骤
