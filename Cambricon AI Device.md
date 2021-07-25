# Research On Cambricon AI Device and Its Development Toolkits
[版权所有:梁德鹏@Ldpe2G](https://github.com/Oneflow-Inc/OneTeam/issues/25)

寒武纪AI硬件以及软件栈调研，持续更新。


## [寒武纪硬件产品](http://www.cambricon.com/)

从产品的介绍来看，不管是边缘端还是云端，目前寒武纪的硬件主要是**面向推理**场景，主要宣传的是`定点运算`的性能。

### 边缘计算，[思元220系列](http://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=57)

思元220芯片基于寒武纪MLUv02架构，标准M.2加速卡集成了 **8TOPS(INT8)** 理论峰值性能，功耗仅为8.25W，可以轻松实现终端设备和边缘端设备的AI赋能方案。

### 云端推理

- #### [思元100系列](http://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=21)

  思元100是寒武纪推出的第一款智能处理板卡产品，基于寒武纪MLUv01架构。

  思元100加速卡的 INT8 理论峰值性能为32TOPS，在稀疏模式下等效理论峰值性能为 128TOPS，FP16 理论峰值性能为16TFLOPs，稀疏模式下等效理论峰值性能为64TFLOPs。

  思元100加速卡还搭载多种容量的256bit DDR4 ECC内存，可满足各类推理场景的云端计算需求。

  ![image-20200825101712117](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200825101712117.png)



- #### [思元270系列](http://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=15)

  思元270系列采用寒武纪MLUv02架构，处理非稀疏深度学习模型的理论峰值性能提升至上一代思元100的4倍，达到 128TOPS(INT8)；

  同时兼容INT4和INT16运算，理论峰值分别达到256TOPS和64TOPS；

  支持浮点运算和混合精度运算。

   - ##### [思元270-S4](http://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=36)

     为高能效比AI推理设计的数据中心级加速卡。

     ![image-20200825103419390](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200825103419390.png)

   - [思元270-F4](http://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=37)

     面向非数据中心AI推理，为桌面环境提供数据中心级AI计算力。

     ![image-20200825103901327](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200825103901327.png)



**MLU  硬件架构设计文档：http://www.cambricon.com/docs/bangc/developer_guide_html/3MLUhardwarearchitecture/index.html#hardware-concepts**





## [配套开发工具](http://forum.cambricon.com/list-64-1.html)

[寒武纪端云一体人工智能开发平台白皮书](http://forum.cambricon.com/list-79-1.html)

### [CNMon](http://www.cambricon.com/docs/cnmon/cnmon_overview/cnmon_overview.html)，Cambricon Neuware Monitor，寒武纪硬件监测器工具

CNMon是一款寒武纪硬件检测工具，通过调用CNDev接口获取底层硬件信息。

CNMon不仅可以采集底层硬件信息，还可以实时获取上层软件对硬件资源的开销，为用户实时显示当前底层硬件的详细信息和状态。

![image-20200825123101059](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200825123101059.png)

更多使用方式见：http://www.cambricon.com/docs/cnmon/cnmon_usage/cnmon_usage.html



###  [CNGDB](http://www.cambricon.com/docs/CNGDB/index.html)，调试工具

 - [简介](http://www.cambricon.com/docs/CNGDB/cngdb_public/3_introduction/cngdb_3.html)

   CNGDB 是运行在 Linux 上的 Cambricon 软件调试工具，基于 GNU 开源项目 GDB 二次开发。

   CNGDB 不仅全面兼容 GDB 原生命令，还包含如下特性：

   - 支持在一次运行中同时调试 BANG C 的 MLU 端和 C/C++ 的 Host 端代码。
   - 支持调试 Cambricon 的单核和多核应用程序。
   - 允许用户设置 MLU 硬件断点，单步 Cambricon 应用程序以及检查和修改程序在所运行的 core(s) 上的变量或其他内存数据。
   - 支持 **MLU220**、**MLU270** 及后续版本架构的所有硬件，包括但不限于 **MLU290** 。

- [快速入门](http://www.cambricon.com/docs/CNGDB/cngdb_public/4_usage/index.html)



### [CNRT](http://www.cambricon.com/docs/cnrt/user_guide_html/quickstart/index.html)，Cambricon Neuware Runtime Library，寒武纪运行时库

CNRT 提供了一套面向MLU（`Machine Learning Unit`，寒武纪机器学习单元）设备的高级别的接口，用于主机与MLU设备之间的交互。

[执行下面步骤运行离线模型:](https://www.cambricon.com/docs/cnrt/user_guide_html/programming/programming_guide.html#id1)<br>

```.h
1. 调用 `cnrtInit()` API，初始化设备。
2. 调用 `cnrtLoadModel()` API，加载离线模型。
3. 调用 `cnrtSetCurrentDevice()` API，设置使用的MLU设备。
4. 调用 `cnrtExtractFunction ()` API，提取离线模型中的模型信息。
5. 通过 `cnrtGetInputDataSize()` 和 `cnrtGetOutputDataSize()` API 获得输入和输出数据的内存大小。
6. 调用 `cnrtMalloc()` API，为MLU输入数据分配内存指定空间。
7. 调用 `cnrtMemcpy()` API，同步拷贝主机端数据到MLU端。
8. 调用 `cnrtMalloc()` API，为MLU输出数据分配内存指定空间。
9. 设置Context。
   1. 调用 `cnrtCreateRuntimeContext()` API，创建Context。
   2. 调用 `cnrtSetRuntimeContextDeviceId()` API，绑定设备。
   3. 调用 `cnrtInitRuntimeContext()` API，初始化Context。
   4. 调用 `cnrtRuntimeContextCreateQueue()` API，创建队列。
10. 调用 `cnrtInvokeRuntimeContext()` API，将任务下发到队列。
11. 调用 `cnrtSyncQueue()` API，同步任务。
12. 调用 `cnrtMemcpy()` API，将计算结果从MLU拷出到CPU。
```

[离线模型示例程序](https://www.cambricon.com/docs/cnrt/user_guide_html/example/offline_mode.html#offlinesample)<br>
下面示例程序是全连接（mlp）算子的离线模型加载及计算过程。<br>
```.h
/* Copyright (C) [2019] by Cambricon, Inc. */
  /* offline_test */
  /*
   * A test which shows how to load and run an offline model.
   * This test consists of one operation --mlp.
   *
   * This example is used for MLU270 and MLU220.
   *
   */

  #include "cnrt.h"
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  int offline_test(const char *name) {

// This example is used for MLU270 and MLU220. You need to
       choose the corresponding offline model.
    // when generating an offline model, u need cnml and cnrt both
    // when running an offline model, u need cnrt only
    cnrtInit(0); // 1. 调用 `cnrtInit()` API，初始化设备。

    // prepare model name
    char fname[100] = "../";
    // The name parameter represents the name of the offline model file.
    // It is also the name of a function in the offline model file.
    strcat(fname, name);
    strcat(fname, ".mef");
    // load model
    cnrtModel_t model;
    cnrtLoadModel(&model, fname); // 2. 调用 `cnrtLoadModel()` API，加载离线模型。

    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev); // 3. 调用 `cnrtSetCurrentDevice()` API，设置使用的MLU设备。

    // get model total memory
    int64_t totalMem;
    cnrtGetModelMemUsed(model, &totalMem);
    printf("total memory used: %ld Bytes\n", totalMem);
    // get model parallelism
    int model_parallelism;
    cnrtQueryModelParallelism(model, &model_parallelism);
    printf("model parallelism: %d.\n", model_parallelism);

    // load extract function
    cnrtFunction_t function;
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, name); // 4. 调用 `cnrtExtractFunction ()` API，提取离线模型中的模型信息。

    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function); // 5. 通过 `cnrtGetInputDataSize()` 和 `cnrtGetOutputDataSize()` API 获得输入和输出数据的内存大小。
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function); 

    // prepare data on cpu
    void **inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
    void **outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));

    // allocate I/O data memory on MLU
    void **inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
    void **outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));

    // prepare input buffer
    for (int i = 0; i < inputNum; i++) {
          // converts data format when using new interface model
          inputCpuPtrS[i] = malloc(inputSizeS[i]);
          // malloc mlu memory
          cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]); // 6. 调用 `cnrtMalloc()` API，为MLU输入数据分配内存指定空间。
          cnrtMemcpy(inputMluPtrS[i], inputCpuPtrS[i], inputSizeS[i], CNRT_MEM_TRANS_DIR_HOST2DEV); // 7. 调用 `cnrtMemcpy()` API，同步拷贝主机端数据到MLU端。
    }
    // prepare output buffer
    for (int i = 0; i < outputNum; i++) {
          outputCpuPtrS[i] = malloc(outputSizeS[i]);
          // malloc mlu memory
          cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]); // 8. 调用 `cnrtMalloc()` API，为MLU输出数据分配内存指定空间。
    }

    // prepare parameters for cnrtInvokeRuntimeContext
    void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
    for (int i = 0; i < inputNum; ++i) {
          param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; ++i) {
          param[inputNum + i] = outputMluPtrS[i];
    }

    // setup runtime ctx
    cnrtRuntimeContext_t ctx;
    cnrtCreateRuntimeContext(&ctx, function, NULL); //  9-1. 调用 `cnrtCreateRuntimeContext()` API，创建Context。

    // bind device
    cnrtSetRuntimeContextDeviceId(ctx, 0); // 9-2. 调用 `cnrtSetRuntimeContextDeviceId()` API，绑定设备。
    cnrtInitRuntimeContext(ctx, NULL); // 9-3. 调用 `cnrtInitRuntimeContext()` API，初始化Context。

    // compute offline
    cnrtQueue_t queue;
    cnrtRuntimeContextCreateQueue(ctx, &queue); // 9-4. 调用 `cnrtRuntimeContextCreateQueue()` API，创建队列。

    // invoke
    cnrtInvokeRuntimeContext(ctx, param, queue, NULL); // 10. 调用 `cnrtInvokeRuntimeContext()` API，将任务下发到队列。

    // sync
    cnrtSyncQueue(queue); // 11. 调用 `cnrtSyncQueue()` API，同步任务。

    // copy mlu result to cpu
    for (int i = 0; i < outputNum; i++) {
          cnrtMemcpy(outputCpuPtrS[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST); // 12. 调用 `cnrtMemcpy()` API，将计算结果从MLU拷出到CPU。
    }

    // free memory space
    for (int i = 0; i < inputNum; i++) {
          free(inputCpuPtrS[i]);
          cnrtFree(inputMluPtrS[i]);
    }
    for (int i = 0; i < outputNum; i++) {
          free(outputCpuPtrS[i]);
          cnrtFree(outputMluPtrS[i]);
    }
    free(inputCpuPtrS);
    free(outputCpuPtrS);
    free(param);

    cnrtDestroyQueue(queue);
    cnrtDestroyRuntimeContext(ctx);
    cnrtDestroyFunction(function);
    cnrtUnloadModel(model);
    cnrtDestroy();

    return 0;
  }

  int main() {
    printf("mlp offline test\n");
    offline_test("mlp");
    return 0;
  }
```


### [CNStream](http://www.cambricon.com/docs/cnstream/user_guide_html/quickstart/quickstart.html)，数据流处理SDK

​	https://github.com/Cambricon/CNStream

- [CNStream详细概念和功能介绍](http://www.cambricon.com/docs/cnstream/user_guide_html/overview/Overview.html#overview)

  CNStream能够大大简化寒武纪深度学习平台提供的推理和其他处理，如视频解码、神经网络图像前处理的集成。

  CNStream基于模块化和流水线的思想，提供了一套基于C++语言的接口来支持流处理多路并发的Pipeline框架。

  为用户提供可自定义的模块机制以及通过高度抽象的 [CNFrameInfo](http://www.cambricon.com/docs/cnstream/developer_guide_html/cnstream_api/data/frame.html#cnframeinfo) 类型进行模块间的数据传输，满足用户对性能和可伸缩性的需求。

  CNStream支持在**MLU270和MLU220 M.2**平台上使用。

- [安装和配置环境依赖和依赖库](http://www.cambricon.com/docs/cnstream/user_guide_html/quickstart/quickstart.html)
- [Developer Guide](http://www.cambricon.com/docs/cnstream/developer_guide_html/)

#18 




# BANG C，开发语言简介(梁德鹏@Ldpe2G)
# [BANG C](http://www.cambricon.com/docs/bangc/developer_guide_html/2Overview/index.html)，开发语言简介

BANG C 是寒武纪开发的针对 MLU 硬件的开发语言，如果后续支持寒武纪MLU的话，这块是需要重点关注的。

![image-20200825121720146](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200825121720146.png)

BANG C 是一门基于 C/C++ 的扩展语言，然后用 BANG C 编程的 MLU 程序可以用 [CNCC](http://www.cambricon.com/docs/bangc/developer_guide_html/2Overview/index.html#f1-1) 编译器来编译。

从上图可以看到，Host 端的 C/C++ 代码通过 GCC 或者 CLANG 编译器来编译，最后 Host 端的 linker 再把两端的 `.o` 文件链接成可执行文件。

下面是把 l2loss.mlu, l2loss_main.cpp 和 l2loss_ops.cpp 三个源文件编译成 l2loss 可执行文件的例子：

```bash
cncc -c l2loss.mlu --bang-device-only -o l2loss.o
g++ -c l2loss_main.cpp
g++ -c l2loss_ops.cpp  -I/usr/local/cambricon/include
g++ l2loss.o l2loss_main.o l2loss_ops.o -o l2loss -L/usr/local/cambricon/lib -lcnrt
```

### Host 端代码

Host 端就是只通常的  C/C++程序，通过调用 CNRT API 初始化设备，然后管理设备内存，准备 Kernel 的参数，启动 Kernel ，最后释放资源。

下面就简单介绍如何在 Host 端调用 Kernel 程序。

首先 Host 端的代码需要 `include cnrt.h` 头文件， 接着在启动 Kernel 之前用需要初始化设备：

```C++
cnrtInit(0);
```

在初始化设备之后，用户可以通过下面的代码获取并设置设备：

```C++
cnrtDev_t dev;
cnrtGetDeviceHandle(&dev, 0);
cnrtSetCurrentDevice(dev);
```

接着用户可以调用 `cnrtConvertDoubleToHalf/cnrtConvertFloatToHalf`函数来得到半精度的输入数据:

```C++
typedef uint16_t half;
half* input_half = (half*)(malloc(dims_a * sizeof(half)));
for (int i = 0; i< len; i++) {
  cnrtConvertFloatToHalf(input_half+i, input[i]);
}
```

然后用户调用 `cnrtMalloc` 分配设备上的空间，接着调用  `cnrtMemcpy` 把数据拷贝到设备上：

```C++
half* mlu_input;
cnrtMalloc((void**)(&mlu_input), dims_a * sizeof(half));
cnrtMemcpy(mlu_input, input_half, dims_a * sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV);
```

然后用下面的代码准备 Kernel 的参数：

```C++
cnrtKernelParamsBuffer_t params;
cnrtGetKernelParamsBuffer(&params);
cnrtKernelParamsBufferAddParam(params, &mlu_input, sizeof(half*));
```

用户在启动一个Kernel 之前需要明确该 Kernel 再哪个队列上执行：

```C++
cnrtQueue_t pQueue;
cnrtCreateQueue(&pQueue);
```

### [关于 Queue ](http://www.cambricon.com/docs/bangc/developer_guide_html/4ProgrammingModel/index.html#queue)

BANG C 提供了一个 类似  cuda stream 的概念 queue：

- 发送到 queue 里面的 Kernel 是异步执行的； 
- 同一个 queue 里面的任务是按照入队的顺序执行的；
- 不同 queue 中的任务是并行执行的。

接着用户需要指定 Kernel 任务的 size。更多关于 size 的设置可以参考：http://www.cambricon.com/docs/bangc/developer_guide_html/4ProgrammingModel/index.html#parallel-model

```c++
cnrtDim3_t dim;
dim.x = 1;
dim.y = 1;
dim.z = 1;
```

设置 kernel task 的类型为 `CNRT_FUNC_TYPE_BLOCK`， 然后启动 Kernel ：

```C++
cnrtFunctionType_t ft = CNRT_FUNC_TYPE_BLOCK;
ret = cnrtInvokeKernel_V2((void *)(&L2LossKernel), dim, params, ft, pQueue);
```

### [关于task 类型](http://www.cambricon.com/docs/bangc/developer_guide_html/4ProgrammingModel/index.html#task-type)

- #### Union(CNRT_FUNC_TYPE_UNIONn)

  表示这是这一个多核并行任务。UNIONn 表示该任务会占用 n 个 硬件 CLUSTERs。

  MLU 270 提供以下几种并行任务类型：

  - UNION1：1 个硬件 CLUSTER，4 核 .
  - UNION2：2 个硬件 CLUSTER，8 核.
  - UNION4：4 个硬件 CLUSTER，16 核.

- #### Block(CNRT_FUNC_TYPE_BLOCK)

  Block 表示单核任务的硬件核的抽象。表示只有一个硬件核心被该任务占用。MLU 270 可以同时调度 32 个不同用户的 Block 任务。

计算任务完成之后，把数据拷贝回 Host 端，并释放资源：

```c++
ret = cnrtSyncQueue(pQueue);
cnrtMemcpy(output_half, mlu_output, sizeof(half), CNRT_MEM_TRANS_DIR_DEV2HOST);
cnrtConvertHalfToFloat(output, output_half[0]);
ret = cnrtDestroyKernelParamsBuffer(params);
ret = cnrtDestroyQueue(pQueue);
cnrtFree(mlu_input);
free(output_half);
cnrtDestroy();
```

#### BANG C Kernel 代码例子

下面代码展示了如何编写 MLU 的 L2Loss Kernel：

```C++
#include "mlu.h"
#define ONELINE 64
__mlu_entry__ void L2LossKernel(half* input, half* output, int32_t len) {
  // __nram__ represents the neuron ram on the chip
  // 关于更多地址空间的修饰符可以参考文档
  // http://www.cambricon.com/docs/bangc/developer_guide_html/6SpecificationforBANGC/index.html?highlight=gdram
  __nram__ int32_t quotient = len / ONELINE;
  __nram__ int32_t rem = len % ONELINE;
  __nram__ half input_nram[ONELINE];
  output[0] = 0;
  for (int32_t i = 0; i < quotient; i++) {
    // 在进行向量运算之前需要把数据从  GDRAM 移动到 NRAM
    __memcpy(input_nram, input + i * ONELINE, ONELINE * sizeof(half) , GDRAM2NRAM);
    // __bang_mul 对两个向量做eltwise乘法运算
    // 要求向量元素个数乘以 sizeof(元素类型) 要能整除 128;
    // 详细文档见：http://www.cambricon.com/docs/bangc/developer_guide_html/13Built-inFuction/index.html?highlight=__bang_mul 
    __bang_mul(input_nram, input_nram, input_nram, ONELINE);
    __bang_mul_const(input_nram, input_nram, 0.5, ONELINE);
    for (int32_t j = 0; j < ONELINE; j++) {
      output[0] += input_nram[j];
    }
  }
  // 处理尾部数据
  if (rem != 0) {
    __memcpy(input_nram, input + quotient * ONELINE,
      ONELINE * sizeof(half), GDRAM2NRAM);
    __bang_mul(input_nram, input_nram, input_nram, ONELINE);
    __bang_mul_const(input_nram, input_nram, 0.5, ONELINE);
    for (int i = 0; i < rem; i++) {
      output[0] += input_nram[i];
    }
  }
}
```

从上面的例子来看，当  kernel 类型为 `CNRT_FUNC_TYPE_BLOCK` 单核模式的时候 MLU 的加速方式有点像 cpu 端比如 x86 的 SSE 或者 Arm 的 NEON 向量指令的加速方式。

### 并行编程的例子

BANG C 语言提供了许多内置的变量来让用户显式进行并行编程。

关于内置变量，可以参考文档：http://www.cambricon.com/docs/bangc/developer_guide_html/4ProgrammingModel/index.html#parallel-built-in-variable 

### 任务分割

下面代码例子是，把任务类型设置为 CNRT_FUNC_TYPE_UNION4，在x维度分成 16 份：

```
cnrtFunctionType_t Union4 = CNRT_FUNC_TYPE_UNION4;
Dim3_t dim;
dim.x = 16;
dim.y = 1;
dim.z = 1;
cnrtInvokeKernel_V2((void *)&add, taskDim, params, Union4, pQueue);
```

任务的内置变量包括 taskDim, taskDimX, taskDimY, taskDimZ, taskIdX, taskIdY, taskIdZ, taskId 。

每个 Kernel 运行时会由多个任务组成，然后每个任务会被映射到一个计算核上，且可以通过 `Dim3_t` 数据结构来索引。

```C++
taskId = taskIdZ * taskDimY * taskDimX + taskIdY * taskDimX + taskIdX
```

下面以数组加为例，数组 x 和 y 的长度分别是 16384 ，然后假设任务类型设置为  Union4，dim.x = 16。则每个任务负责 1024 个元素的加法运算：

```C++
#define N 1024
__mlu_entry__ void add(float* x, float* y, float*z ) {
    __nram__ float x_tmp [N];
    __nram__ float y_tmp [N];
    __memcpy ( x_tmp , x + taskId * N, N * sizeof (float) , GDRAM2NRAM);
    __memcpy ( y_tmp , y + taskId * N, N * sizeof (float) , GDRAM2NRAM);
    __bang_add ( x_tmp, x_tmp, y_tmp, N) ;
    __memcpy ( z + taskId * N, x_tmp , N * sizeof (float) , NRAM2GDRAM);
}
```

更多关于 bang c 编程概念：http://www.cambricon.com/docs/bangc/developer_guide_html/index.html



## [如何调试 BAND C 代码](http://www.cambricon.com/docs/bangc/developer_guide_html/10BANGProgramDebugging/index.html)

## [BAND C 程序性能优化指导](http://www.cambricon.com/docs/bangc/developer_guide_html/9PerformanceGuide/index.html)



# 梁德鹏@Ldpe2G
## CNML 库简介

![soft_group](https://user-images.githubusercontent.com/31394900/126887041-879b2cc6-6fef-4669-877e-b9334c8ecfc6.png)




上层的机器学习应用可以直接采用各种编程框架的编程接口，如 TensorFlow、Caffe、Caffe2、MXNet 等,间接通过 CNML 调用 CNRT 进行软件编程，或者直接用 BANG C 编写算子 Kernel 。



看了下 MXNet，TensorFlow 和 Pytorch `添加 寒武纪 支持的文档`，发现都是用的寒武纪提供的 `CNML 库（类似 CUDNN）`。

CNML 是针对机器学习以及深度学习的编程库，⽤于在 MLU 上加速客⼾各种机器学习/深度学习算法。

而 CNML 在官网上并没有提供给相关使用文档，只在袁老师给的寒武纪的资料里面有。



⽬前通过 CNML 基本算⼦及其组合出来的算⼦已经超过 140 个，而且用户还可以根据⾃⼰的需求⾃由增加算⼦。

不过 CNML 目前只提供了算子的 forward 函数，并没有提供 backward ，所以目前还是只能做 推理 加速。



 下面代码示例介绍了如何运⽤ CNML 和 CNRT 库创建⼀个简单的 add 算⼦的过程：

```
1. 创建 CPU 端的张量，并准备 CPU 端的数据。
2. 创建 MLU 端的张量, 利⽤ MLU 端的张量做为输⼊创建 AddOp。
3. 编译。
4. 将 CPU 端的数据拷⼊到 MLU 端。
5. 执⾏ MLU 端的计算过程。
6. 将 MLU 端的结果拷⻉到 CPU 端。
7. 计算结果存放在⽂件中。
8. 释放 MLU 和 CPU 端的资源。
```

```C++
/*
 * op name: add
 * input_1 size: n x c x h x w
 * input_2 size: n x c x h x w
 * output size: n x c x h x w
 */

// CNML 的 Tensor 数据结构只是⽤于描述参与运算的操作数的维度、类型、数据类型、存储顺序等信息，并不存数据。

int add_test() {
    // this demo used for MLU270 and MLU220. If you are using MLU220, set coreVersion to CNML_MLU220.
    const cnmlCoreVersion_t coreVersion = CNML_MLU270;
    // using cores num. Set the core numbers based on MLU270 or MLU220 you are using.
    const int coreNum = 4;
    // prepare data for pool
    const int dimNum = 4;
    const int n = 1, c = 32, h = 4, w = 4;
    // count input, filter, bias, output nums
    int input_count_1 = n * h * w * c;
    int input_count_2 = n * h * w * c;
    int output_count = n * h * w * c;
    float *input_cpu_ptr_1 = (float *)malloc(input_count_1 * sizeof(float));
    float *input_cpu_ptr_2 = (float *)malloc(input_count_2 * sizeof(float));
    float *output_cpu_ptr = (float *)malloc(output_count * sizeof(float));
    unsigned int seed = time(0);

    for (int i = 0; i < input_count_1; i++) {
        input_cpu_ptr_1[i] = ((rand_r(&seed) % 100 / 100.0) - 0.5) / 2;
    }
    for (int i = 0; i < input_count_2; i++) {
        input_cpu_ptr_2[i] = (rand_r(&seed) % 100 / 100.0) - 0.5;
    }
    // set tensor shapes
    int input_shape_1[] = {n, c, h, w};
    int input_shape_2[] = {n, c, h, w};
    int output_shape[] = {n, c, h, w};
    // prepare input tensor 1
    cnmlTensor_t input_tensor_1 = NULL;
    cnmlCreateTensor_V2(&input_tensor_1, CNML_TENSOR);
    cnmlSetTensorShape_V2(input_tensor_1, dimNum, input_shape_1, NULL);
    cnmlSetTensorDataType(input_tensor_1, CNML_DATA_FLOAT32);
    // prepare input tensor 2
    cnmlTensor_t input_tensor_2 = NULL;
    cnmlCreateTensor_V2(&input_tensor_2, CNML_TENSOR);
    cnmlSetTensorShape_V2(input_tensor_2, dimNum, input_shape_2, NULL);
    cnmlSetTensorDataType(input_tensor_2, CNML_DATA_FLOAT32);
    // prepare output tensor
    cnmlTensor_t output_tensor = NULL;
    cnmlCreateTensor_V2(&output_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(output_tensor, dimNum, output_shape, NULL);
    cnmlSetTensorDataType(output_tensor, CNML_DATA_FLOAT32);
    // create add operator
    cnmlBaseOp_t add_op;
    cnmlCreateAddOp(&add_op, input_tensor_1, input_tensor_2, output_tensor);
    // compile op
    cnmlSetBaseOpCoreVersion(add_op, coreVersion);
    cnmlSetBaseOpCorenum(add_op, coreNum);
    cnmlCompileBaseOp_V2(add_op);
    // mlu buffer ptr
    void *input_mlu_ptr_1 = NULL;
    void *input_mlu_ptr_2 = NULL;
    void *output_mlu_ptr = NULL;
    // malloc cnml tensor
    cnrtMalloc(&input_mlu_ptr_1, input_count_1 * sizeof(float));
    cnrtMalloc(&input_mlu_ptr_2, input_count_2 * sizeof(float));
    cnrtMalloc(&output_mlu_ptr, output_count * sizeof(float));
    // copy input to cnml buffer
    cnrtMemcpy(input_mlu_ptr_1, input_cpu_ptr_1, input_count_1 * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(input_mlu_ptr_2, input_cpu_ptr_2, input_count_2 * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);
    // set cnrt queue
    cnrtQueue_t queue;
    cnrtCreateQueue(&queue);
    cnmlComputeAddOpForward_V4(add_op, NULL, input_mlu_ptr_1, NULL, input_mlu_ptr_2, NULL, output_mlu_ptr, queue, NULL);
    // wait for computing task over
    cnrtSyncQueue(queue);
    // end of queue life cycle
    cnrtDestroyQueue(queue);
    // copy output to cpu
    cnrtMemcpy(output_cpu_ptr, output_mlu_ptr, output_count * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST);
    // dump mlu result to file mlu_output
    cnmlDumpTensor2File_V2("mlu_output", output_tensor, output_cpu_ptr, false);
    // delete op ptr
    cnmlDestroyBaseOp(&add_op);
    // delete cnml buffer
    cnrtFree(input_mlu_ptr_1);
    cnrtFree(input_mlu_ptr_2);
    cnrtFree(output_mlu_ptr);
    // delete cnml tensors
    cnmlDestroyTensor(&input_tensor_1);
    cnmlDestroyTensor(&input_tensor_2);
    cnmlDestroyTensor(&output_tensor);
    // delete pointers (including data pointers)
    free(input_cpu_ptr_1);
    free(input_cpu_ptr_2);
    free(output_cpu_ptr);
    return 0;
}
```





## MXNet、 TensorFlow 和 Pytorch  集成 寒武纪 软件栈



### MXNet（详细内容可以参考：Cambricon MXNet⽤⼾⼿册）



MXNet [proporsal](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=120722127) 里面有一章是关于集成 寒武纪 CNML 库的：[Design of CNML/CNRT Integration](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=120722127)



Cambricon MXNet 对原⽣ MXNet 的部分模块进⾏了扩展和修改，如下图阴影部分所⽰：

![image-20200827150113204](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200827150113204.png)



为了能够将 NDArray 和 GraphExecutor 的 Context 设置为 MLU。

Cambricon MXNet 扩展了 struct Context，⽀持设置 MLU 设备、MLU ID 等属性。

Cambricon MXNet 扩展了 NDArray，⽤来管理和使⽤ MLU 上的 Tensor。

Cambricon MXNet 中 MXNet 算⼦的 MLU 实现需要使⽤⼀个或多个 CNML 算⼦来拼接，因此新增 MluOpExecutor 类继承⾃原⽣的 OpExecutor 类，在 MluOpExecutor 类中新增字段 internal_ops_ 和 inter_results_ 分别⽤来存储拼接⽤到的 CNML 算⼦和计算过程中产⽣的中间结果。

为⽀持⽤⼾使⽤符号式模式在线运⾏深度神经⽹络，Cambricon MXNet 新增 MluFuseGraphExecutor 类继承⾃原⽣的 GraphExecutor 类，在 MLuFuseGraphExecutor 类中提供 MluFuseSimpleBind ⽅法实现计算图融合段划分、各融合段的 FusionOp 初始化，同时提供 MluFuseRunOps ⽅法实现各融合段的编译执⾏。

为了多设备并⾏执⾏神经⽹络，Cambricon MXNet 扩展了 Dependency Engine，⽀持使⽤ ThreadedEngine 实现多设备并⾏。



![image-20200827152408692](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200827152408692.png)



更多内容可以参考 第8章。



### TensorFlow（详细内容可以参考：Cambricon TensorFlow ⽤⼾⼿册）

Cambricon TensorFlow 对原始 TensorFlow 代码的修改主要包括五个⽅⾯：

- 添加 MLU 设备⽀持

- 调⽤CNML 实现 MLU 操作
- 实现 MLU 特有执⾏模式（例如算⼦融合）
- ⽀持 Python/C/C++ 应⽤程序调⽤MLU 
- 添加 MLU 专属⼯具包。

对 TensorFlow 框架代码的修改如下图阴影部分所⽰：

![image-20200827152538422](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200827152538422.png)

![image-20200827152746326](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200827152746326.png)

更多内容可以参考 第8章。



### Pytorch（详细内容可以参考：Cambricon PyTorch ⽤⼾⼿册）

 Cambricon PyTorch 主要修改有：

- 添加层在 MLU 设备上的⽀持，调⽤ CNML 实现 MLU 操作
- 实现 MLU 特有的执⾏模式（例如算⼦的融合）
- ⽀持 Python/C++ 应⽤程序调⽤ MLU 以及添加 MLU 专属⼯具包
- 新增分类和检测⽹络程序案例，完善运⾏模式



![image-20200827153153046](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200827153153046.png)

![image-20200827153227772](https://gitee.com/Ldpe2G/picgo/raw/master/image-20200827153227772.png)

更多内容可以参考 第8章。

