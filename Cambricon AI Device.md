# Research On Cambricon AI Device and Its Development Toolkits
[版权所有](https://github.com/Oneflow-Inc/OneTeam/issues/25)
寒武纪AI硬件以及软件栈调研，持续更新。


## [寒武纪硬件产品](http://www.cambricon.com/)

从产品的介绍来看，不管是边缘端还是云端，目前寒武纪的硬件主要是**面向推理**场景，主要宣传的是定点运算的性能。

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

CNRT 提供了一套面向MLU（Machine Learning Unit，寒武纪机器学习单元）设备的高级别的接口，用于主机与MLU设备之间的交互。

执行下面步骤运行离线模型：

```C++
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
