# 扩展设备类型的设计讨论

支持不同设备类型框架层面原理图
<img width="876" alt="graph" src="https://user-images.githubusercontent.com/31394900/126856050-ffcfa136-dbf0-41b6-95ad-37ead15aacce.png">


# 成诚 @chengtbf

到底怎样的设计才是灵活支持各种不同设备类型的最好设计？

## 我心中的完美设计

### 1.  Kernel的实现(计算)，内存的逻辑(存储)，boxing/copy的逻辑（传输） 跟 Device 解耦

​	目前的很多Kernel的GPU实现都自己去写CUDA的global kernel，并用 `<<<...>>>` 之类的语法调用这些cuda kernel，或者是调用cudnn的接口等。我们后面会增加越来越多的算子，甚至发展到后期，上千个算子都有可能。我们新支持一个设备类型，不可能再把这上千个算子在新设备类型下的实现重新再写一遍。所以要做到kenel实现跟具体的设备类型无关。

​	所以需要提供一个“Device”的抽象， Device包含了一套基础的算子库的接口，每个具体的Device都需要实现这套基础算子库

- Device 抽象

  - 包含了一个操作符/数据类型的接口集合（加强版的KernelUtil），接口需要传入DeviceContext 和 数据的指针等参数。这个接口集合就是基础的所有计算的集合，任何复杂的op逻辑，都可以由这个接口集合组合得到
  - 包含获取内存的容量/占用情况/申请/释放内存的接口
  - stream handle等
  - copy host to device / copy device to host
  - 集合通信/allreduce/device上的boxing逻辑
  - ...

  那么在kernel的代码里，是不能出现.cu文件，也不能自己再新定义一个`__global__ `  funciton的。

  Kernel的Compute逻辑只能调用Device提供的各种math接口来做计算，这样就做到了Kernel的计算逻辑跟Device类型解耦。

  这样的好处是：**无论有多少个Kernel实现，新的设备类型都不需要关心。每新增一个设备类型，只需要关心Device这个抽象里的接口如何重载实现即可**

- Device目录下提供各种不同的设备类型支持，CUDA，ROCm，... 每个新增的设备类型只需要在Device目录下新增一个子目录，并重载其接口
- 只有Device/CUDA目录下才有.cu文件和一切跟WITH_CUDA相关的逻辑
- 内存申请/释放 逻辑、 数据搬运逻辑 跟Device 类型解耦
- Nccl 的 op/kernel/task node/logical node 等需要删掉
- Boxing要支持不同的设备传输，Nccl只是其中的一个选项

### 2. 只有一个backend —— 每个OneFlow的安装包，只支持一种Device

- DeviceType之外还需要有一个基础概念，描述op/task node的placement，这个变量只有两个值 { kHost, kDevice }

- 编译期只关心两种：Host or Device。Host就是CPU，Device可以是任意的设备（N卡、A卡、TPU、NPU...），所有对op的分析、infer逻辑只需要关心device的逻辑。
- 把编译期所有对CUDA、`DeviceType::kGPU`、`DeviceType::kCPU`等逻辑都需要过一遍，将GPU的逻辑替换为Device
- 如果是CPU Only的安装包，就只有host，如果是N卡设备，就编译oneflow_cu102之类的，如果是A卡设备，就编译一个oneflow_rocm..之类的安装包
- Cmake里的逻辑要处理这种one of的情形，本次cmake是针对哪种设备，WITH_CUDA或者WITH_ROCM等

### 3. Python端的处理

- gpu_device_num -> device num
- Python端对Placement只需要关心是Device，而不需要关心具体是哪种Device

### 4. id manager

​	对Thread Id、Task Id、Device id的分配规则、命名的处理。如果只需要支持一个Devcie的Backend，那么分配规则不变，只需要把GPU换成device就行

## 其他还需要注意的点

### 1. 内存申请/MemCase

  所有对内存的查询、申请/释放等操作需要替换（memory allocator、vm里的cuda allocator等等）

### 2. 对half的处理

​	目前的dtype有float16（对应c++里的class half_float::half），但是在CUDA上是把float16指针转化成了CUDA里的half struct进行操作。这里是把HALF作为一个特殊的DataTypeSeq来处理的。如果要支持其他框架，也需要考虑half类型如何兼容不同的设备

### 3. 单测/集成测试

​	测试里大量使用了gpu的device type做测试，在不同的设备类型下，如何给每个设备类型做对应的测试。需要修改大量的单测脚本。

### 4. Kernel conf/Op InferBlobDesc

-  比如conv op在cudnn下需要infer algo，这个逻辑需要考虑各种不同的设备

- 或者有些只专门支持cudnn的 op，在其他的比如ROCm是另一种op？维护一个device->supported op list的映射？
- 对于同一个op，比如conv，不同的设备，host、device(CUDA、ROCM、Cudnn...) 是否需要的Tmp size不同？或者需要不同的blob name？可能需要特别考虑一下

### 5. Eager那边对gpu/cuda/stream handle/Placement等的处理也需要改



之后再想到什么需要注意的再补充

我认为对于OneFlow而言，只有两种DeviceType { kHost, kDevice }
然后cmake和宏来控制每次编译的时候用哪种后端：CUDA、ROCm ...
对于 complier、runtime、python、op/kernel、mem allocator 等等概念，只区分host和device，不关心具体是哪种backend


# 梁德鹏@Ldpe2G
调研了下其他框架（tvm, mxnet, mindspore, megengine, paddlepaddle, tensorflow, pytorch）对多设备类型的定义方式，初步调研结果：

### 1、TVM 的设备类型直接采用 dlpack 的 [enum](https://github.com/dmlc/dlpack/blob/3ec04430e89a6834e5a1b99471f415fa939bf642/include/dlpack/dlpack.h#L38)：

```C++
typedef enum {
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLGPU = 2,
  /*!
   * \brief Pinned CUDA GPU device by cudaMallocHost
   * \note kDLCPUPinned = kDLCPU | kDLGPU
   */
  kDLCPUPinned = 3,
  /*! \brief OpenCL devices. */
  kDLOpenCL = 4,
  /*! \brief Metal for Apple GPU. */
  kDLMetal = 8,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*!
   * \brief Reserved extension device type,
   * used for quickly test extension device
   * The semantics can differ depending on the implementation.
   */
  kDLExtDev = 12,
} DLDeviceType;
```

kDLGPU 指 cuda，kDLROCM 指 amd gpu， python 前端 用 string 设置设备类型，在C++后端根据注册的映射表得到 [string -> enum 的映射](https://github.com/apache/incubator-tvm/blob/master/src/target/target_kind.cc#L88)，部分示例代码：

```C++
TVM_REGISTER_TARGET_KIND("llvm")
    .set_default_keys({"cpu"})
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_KIND("cuda")
    .set_default_keys({"cuda", "gpu"})
    .set_device_type(kDLGPU);

TVM_REGISTER_TARGET_KIND("rocm")
    .set_default_keys({"rocm", "gpu"})
    .set_device_type(kDLROCM);

TVM_REGISTER_TARGET_KIND("opencl")
    .set_default_keys({"opencl", "gpu"})
    .set_device_type(kDLOpenCL);

TVM_REGISTER_TARGET_KIND("metal")
    .set_default_keys({"metal", "gpu"})
    .set_device_type(kDLMetal);
```

###  2、MXNet 官方目前只支持  [cuda gpu 类型](https://github.com/apache/incubator-mxnet/blob/master/include/mxnet/base.h#L92):

```C++
namespace mshadow {
struct cpu {
  static const bool kDevCPU = true;
  static const int kDevMask = 1 << 0;
};
struct gpu {
  static const bool kDevCPU = false;
  static const int kDevMask = 1 << 1;
};
}
typedef mshadow::cpu cpu;
typedef mshadow::gpu gpu;
enum DeviceType {
    kCPU = cpu::kDevMask,
    kGPU = gpu::kDevMask,
    kCPUPinned = 3,
    kCPUShared = 5,
};
```

MXNet python 前端 会在构造 context 的时候通过 [维护的映射表](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/context.py#L65) 得到 device字符串对应的 DeviceType 枚举值:

```C++
class Context:
    devtype2str = {1: 'cpu', 2: 'gpu', 3: 'cpu_pinned', 5: 'cpu_shared'}
    devstr2type = {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3, 'cpu_shared': 5}
	def __init__(self, device_type, device_id=0):
        self.device_typeid = Context.devstr2type[device_type]
        self.device_id = device_id

def gpu(device_id=0):
    return Context('gpu', device_id)
```

在注册 kernel 具体实现的时候也是采用 DeviceType 模板传参方式。

### 3、paddlepaddle，只支持 [cuda gpu 类型](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/place.h#L26)，采用结构体形式标识设备类型:

```C++
struct CPUPlace {
};

struct CUDAPlace {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}
  inline int GetDeviceId() const { return device; }
  int device;
};

struct CUDAPinnedPlace {
};
```

采用了 [pybind11 的特性 ](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/pybind.cc#L1341)生成与设备类型结构体同名 python类：

```python
py::class_<platform::CUDAPlace>(m, "CUDAPlace", R"DOC()DOC")
      .def("__init__",
           [](platform::CUDAPlace &self, int dev_id) {
               # ..... 
           });

py::class_<paddle::platform::CPUPlace>(m, "CPUPlace", R"DOC()DOC")
      .def(py::init<>());
```

然后 python前端直接[实例化该类](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/test_python_operator_overriding.py#L62)：

```python
places = [fluid.CPUPlace()]
if fluid.core.is_compiled_with_cuda():
    places.append(fluid.CUDAPlace(0))
```

### 4、[Mindspore](https://github.com/mindspore-ai/mindspore/blob/master/mindspore/core/utils/ms_context.h#L40) 支持 cuda 和华为的硬件，C++端用char 数组表示设备类型:

```C++
const char kCPUDevice[] = "CPU";
const char kGPUDevice[] = "GPU";
const char kAscendDevice[] = "Ascend";
const char kDavinciDevice[] = "Davinci";
const std::set<std::string> kTargetSet = {kCPUDevice, kGPUDevice, kAscendDevice, kDavinciDevice};
```

Python前端直接传对应设备字符串到C++后端。

### 5、Megengine 只支持 [cuda](https://github.com/MegEngine/MegEngine/blob/master/src/core/include/megbrain/comp_node.h#L108) ：

```C++
enum class DeviceType {
    //! for "xpu" comp node that would mapped to available cn on
    //! current system
    UNSPEC = 0,
    CUDA = 1,
    CPU = 2,
    MULTITHREAD,
    MAX_DEVICE_ID,
};
```

[前端用 字符串表示 device](https://github.com/MegEngine/MegEngine/blob/master/python_module/megengine/core/device.py#L36)：

```python
def set_default_device(device: str = "xpux"):
    r"""Sets default computing node.

    :param device: default device type. The type can be 'cpu0', 'cpu1', etc.,
        or 'gpu0', 'gpu1', etc., to specify the particular cpu or gpu to use.
        'cpux' and  'gupx' can also be used to specify any number of cpu or gpu devices.

        'multithread' device type is avaliable when inference, which implements
        multi-threading parallelism at the operator level. For example,
        'multithread4' will compute with 4 threads. which implements

        The default value is 'xpux' to specify any device available.

        It can also be set by environmental variable `MGE_DEFAULT_DEVICE`.
    """
    global _default_device  # pylint: disable=global-statement
    _default_device = device
```

后端有个[解析函数](https://github.com/MegEngine/MegEngine/blob/master/src/core/impl/comp_node/comp_node.cpp#L121)，得到字符串对应的 enum class ：

```C++
CompNode::Locator CompNode::Locator::parse(const std::string &id) {
	// ......
    DeviceType dev_type;
    // parse dev_type
    if (ptr[0] == 'c') {
        dev_type = DeviceType::CPU;
    } else if (ptr[0] == 'g') {
        dev_type = DeviceType::CUDA;
    }
    return {dev_type, num_dev, {num_stream}};
}
```

### 6、tensorflow 支持多种gpu后端和tpu

根据官方文档，多个gpu后端 [cuda, rocm, opencl之间是互斥的](https://github.com/tensorflow/tensorflow/blob/master/configure.py#L1536)，在编译的时候只能选择其中一个：

```python
  # SYCL / ROCm / CUDA are mutually exclusive.
  # At most 1 GPU platform can be configured.
  gpu_platform_count = 0
  if environ_cp.get('TF_NEED_OPENCL_SYCL') == '1':
    gpu_platform_count += 1
  if environ_cp.get('TF_NEED_ROCM') == '1':
    gpu_platform_count += 1
  if environ_cp.get('TF_NEED_CUDA') == '1':
    gpu_platform_count += 1
  if gpu_platform_count >= 2:
    raise UserInputError('SYCL / CUDA / ROCm are mututally exclusive. '
                         'At most 1 GPU platform can be configured.')
```

然后 对于设备类型的设计，C++后端是[字符串](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.h#L74)和[结构体定义](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/argmax_op.cc#L230)：

```C++
// Convenient constants that can be passed to a DeviceType constructor
TF_EXPORT extern const char* const DEVICE_DEFAULT;     // "DEFAULT"
TF_EXPORT extern const char* const DEVICE_CPU;         // "CPU"
TF_EXPORT extern const char* const DEVICE_GPU;         // "GPU"
TF_EXPORT extern const char* const DEVICE_SYCL;        // "SYCL"
TF_EXPORT extern const char* const DEVICE_TPU_SYSTEM;  // "TPU_SYSTEM"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace Eigen {
struct ThreadPoolDevice;
struct GpuDevice;
struct SyclDevice;
}  // end namespace Eigen
```

这里 DEVICE_GPU 指代 cuda 和 rocm 实现。

### 7、pytorch 支持 cuda, rocm(hip)，opencl，opengl 等多种 gpu backend

[设备类型定义](https://github.com/pytorch/pytorch/blob/master/c10/core/DeviceType.h#L15)：

```C++
enum class DeviceType : int16_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  MSNPU = 8, // MSNPU
  XLA = 9, // XLA / TPU
  Vulkan = 10, // Vulkan
  COMPILE_TIME_MAX_DEVICE_TYPES = 11,
  ONLY_FOR_TEST = 20901, // This device type is only for test.
};
```

python 前端传字符串，C++后端有个[解析转换函数](https://github.com/pytorch/pytorch/blob/master/c10/core/Device.cpp#L32) 得到 device 字符串对应的 enum class：

```C++
DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<std::string, DeviceType>, 10> types = {{
      {"cpu", DeviceType::CPU},
      {"cuda", DeviceType::CUDA},
      {"mkldnn", DeviceType::MKLDNN},
      {"opengl", DeviceType::OPENGL},
      {"opencl", DeviceType::OPENCL},
      {"ideep", DeviceType::IDEEP},
      {"hip", DeviceType::HIP},
      {"fpga", DeviceType::FPGA},
      {"msnpu", DeviceType::MSNPU},
      {"xla", DeviceType::XLA},
  }};
```

看github上的 issue  cuda 和 rocm 等 gpu backend 之间也是互斥的。



## 调研初步结论

1、训练框架中，目前只有 Tensorflow 和 Pytorch 支持了 ROCm；

2、Tensorflow 在底层 kernel实现上，对于cuda 和 rocm 统一用 GPUDevice 表示， kernel 注册部分只区分了 CPUDevice 和 GPUDevice，比如这里用 [crop_and_resize_op](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/crop_and_resize_op.cc#L185) 举例，该 op 对应的的 GPUDevice [模板实现里面](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/crop_and_resize_op_gpu.cu.cc#L376)并没有看到区分当前是什么 backend（cuda 还是 rocm） 的代码，而在其调用的 [GpuLaunchKernel 函数](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/gpu_kernel_helper.h#L100)里面再根据宏来区分当前是  cuda 还是 rocm的。

> 我认为对于OneFlow而言，只有两种DeviceType { kHost, kDevice }
> 然后cmake和宏来控制每次编译的时候用哪种后端：CUDA、ROCm ...
> 对于 complier、runtime、python、op/kernel、mem allocator 等等概念，只区分host和device，不关心具体是哪种backend

感觉 @chengtbf  上面的想法应该是想往这个方向上去做，但是并不局限于 gpu 设备，而是所有异构设备的统一。

3、继续调研，再补充。。。



[DeviceContext](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/device/device_context.h#L34) 基类是否需要重构

```C++
class DeviceCtx {
 public:
 // .......
#ifdef WITH_CUDA
  virtual const cudaStream_t& cuda_stream() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmh_handle() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmd_handle() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_tensor_op_math_handle() const { UNIMPLEMENTED(); }
  virtual const cudnnHandle_t& cudnn_handle() const { UNIMPLEMENTED(); }
#endif

  virtual void SyncDevice() { UNIMPLEMENTED(); }
  virtual void AddCallBack(std::function<void()>) const { UNIMPLEMENTED(); }
};
```

现在是用宏控制是否有cuda相关的函数，后续如果添加新的设备比如 ROCm，目前我能想到的：

1、沿用原来的设计，添加 WITH_ROCM 的宏，把相关函数放宏里，但是如果添加更多的设备，基类可能看起来会臃肿；
2、把所有具体设备相关的函数放子类，比如 cuda 相关的放 cudaDeviceCtx，rocm 的放 rocmDeviceCtx，但是这样子我看了下代码，感觉需要改动很大，因为目前都是以 DeviceCtx 基类来传递，如果这样子改动，则在具体用到 设备相关的函数时候需要 downcast 到子类？感觉 runtime overhead 也不可接受。
 

# 成诚 @chengtbf
> [DeviceContext](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/device/device_context.h#L34) 基类是否需要重构
> 
> ```c++
> class DeviceCtx {
>  public:
>  // .......
> #ifdef WITH_CUDA
>   virtual const cudaStream_t& cuda_stream() const { UNIMPLEMENTED(); }
>   virtual const cublasHandle_t& cublas_pmh_handle() const { UNIMPLEMENTED(); }
>   virtual const cublasHandle_t& cublas_pmd_handle() const { UNIMPLEMENTED(); }
>   virtual const cublasHandle_t& cublas_tensor_op_math_handle() const { UNIMPLEMENTED(); }
>   virtual const cudnnHandle_t& cudnn_handle() const { UNIMPLEMENTED(); }
> #endif
> 
>   virtual void SyncDevice() { UNIMPLEMENTED(); }
>   virtual void AddCallBack(std::function<void()>) const { UNIMPLEMENTED(); }
> };
> ```
> 
> 现在是用宏控制是否有cuda相关的函数，后续如果添加新的设备比如 ROCm，目前我能想到的：
> 
> 1、沿用原来的设计，添加 WITH_ROCM 的宏，把相关函数放宏里，但是如果添加更多的设备，基类可能看起来会臃肿；
> 2、把所有具体设备相关的函数放子类，比如 cuda 相关的放 cudaDeviceCtx，rocm 的放 rocmDeviceCtx，但是这样子我看了下代码，感觉需要改动很大，因为目前都是以 DeviceCtx 基类来传递，如果这样子改动，则在具体用到 设备相关的函数时候需要 downcast 到子类？感觉 runtime overhead 也不可接受。

DeviceContext 肯定需要重构。在我的设想里，Kernel不应该关心CUDA或者ROCm的事，所以类似cuda_stream的接口不需要在DeviceContext的基类里出现。如果我们有个Device抽象（或者叫 DeviceComputeLibrary）用于封装所有Kernel在设备上的计算逻辑，做到Kernel跟Device解耦，Kernel只需要调用Device抽象的计算接口，传入DeviceContext和Tensor的指针就行。具体的Device的子类，如CUDADevice或者ROCmDevice里去dynamic cast成CUDADeviceContext，ROCmDeviceContext，从中获取具体的stream、handle。

所有跟设备相关的代码都需要改。所有的Kernel的GPU实现都需要改。这个应该是必须的吧。

根据俊丞兄在 #16 中的comment https://github.com/Oneflow-Inc/OneTeam/issues/16#issuecomment-678930953

我们编译期还是需要知道具体的DeviceType的，而且可能Kernel跟Device完全分离也有问题。我们需要好好想一下是否需要分两级，即第
是 { host, device }， 第二级是 { CUDA, ROCm, ... } （仅在第一级为device时才有第二级）。类似TF和PyTorch的设计，仍然只有一个DeviceType包含了CPU、各种GPU等，即只有一级。

我觉得即使不做HostKernel和DeviceKernel（即Kernel跟DeviceType解耦），以及即使不做Placement和DeviceType的两层分级。我们也应该让添加一个新的设备类型所需要改动的地方尽可能少，而且尽可能在相同的目录下做更改，而不是每增加一个设备类型，就要改各个地方。 

另， 如果要支持异构的网络（同时有多个设备类型），不同的设备类型的op在传数据的时候应该都要借助host来实现。比如用CUDA to host， host to ROCm 的方式。每个新的设备类型，除了增加对应的计算库，也要支持跟Host之间的Copy操作等。


# 梁德鹏@Ldpe2G
> 我们编译期还是需要知道具体的DeviceType的，而且可能Kernel跟Device完全分离也有问题。我们需要好好想一下是否需要分两级，即第一级是 { host, device }， 第二级是 { CUDA, ROCm, ... } （仅在第一级为device时才有第二级）。类似TF和PyTorch的设计，仍然只有一个DeviceType包含了CPU、各种GPU等，即只有一级。

假如后续支持了异构网络，用户应该是可以自行设置 placement 的，就是网络哪部分放哪类设备上，那这样子是不是不需要分两级。

感觉没有必要分为两级，因为CPU在作为计算设备的时候和GPU的抽象层次应该是相同的


# 成诚 @chengtbf
同意不分两级。一开始设计的两级结构，是为了让python端跟编译期对具体的DeviceType无感，从而减少新增一个新的DeviceType的工作量。现在看来，python端和编译期必须要获得具体的DeviceType，所以分两级就没意义了。  

那么结论应该是我们的**DeviceType还是一级结构**，可能是类似PyTorch的设计，保持c++端是一个枚举类型。

# 李雨芮 @poohRui 
想问一下，基础的算子库的范围应该怎么确定，怎么保证基础算子之外的算子总是可以通过基础算子组合得到？

对于多设备我之前有一个小发现，不过这种属于特例了，像目前有一些用sbatch调度资源的计算平台，如果安装了某框架gpu版本，但是先不想占用GPU资源，只在CPU上做一些验证，这个时候Mindspore会报找不到CUDA相关库的错误，而PyTorch和Tensorflow就支持这种操作，或者说有没有必要考虑这种安装了非CPU版本，但是想在CPU上执行计算的操作。

@chengtbf 说的 ”Kernel只需要调用Device抽象的计算接口，传入DeviceContext和Tensor的指针就行。具体的Device的子类，如CUDADevice或者ROCmDevice里去dynamic cast成CUDADeviceContext，ROCmDeviceContext，从中获取具体的stream、handle。“ 如果提前把每个DeviceContext子类的static实例和对应的设备id存在一个注册表里，应该不需要dynamic cast也能做到，省去一些开销吧。

其实以C++的库为例，在make成功生成了静态/动态库以后，外部用户调用库，使用库里实现的函数，以add函数来说，如果用户只是调用add_cpu版本，库内部不应该进行任何cuda相关的函数调用，尽管编译的cuda版本。我看代码我们现在的多设备基本是靠条件编译来管理的，之前了解Pytorch一部分的设计应该是用注册表的机制把和设备相关的操作与上层调用解耦了，这个设计个人感觉更好看一点。

# 成诚 @chengtbf
> 想问一下，基础的算子库的范围应该怎么确定，怎么保证基础算子之外的算子总是可以通过基础算子组合得到？
> 

我记得华为实现了69个基础算子，就可以拼出来其他所有op了。PyTorch 的Tensor实现了305个基础算子，用于组合生成所有op。 但也没有一个数学上的保证，一个算子集合可以组合出所有的可能的算子。 我之前对于Device的抽象，是想把Kernel跟Device解耦，但跟俊丞兄的讨论中发现可能还是做不到完全解耦，那这个Device的抽象的设计可能就需要重做了。

我觉得可以有一个折中的方案，Device抽象提供一些非常基础和常见的接口，用于实现大部分kernel，对于不能包含其中的kernel，仍用之前的cuda kernel、rocm kernel的方式提供各自的kernel实例。

 
