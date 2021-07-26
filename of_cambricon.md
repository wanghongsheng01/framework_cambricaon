# oneflow_cambricon

版权所有 oneflow support cambricon <br>
[Diff between master and cambricon](https://github.com/Oneflow-Inc/oneflow_cambricon/pull/17/files)


修改的文件
1. 补充 .github/workflows/test.yml 
`github.base_ref == 'cambricon'`

2. CMake 文件中添加 fakedevice 的编译选项<br>

CMakeLists.txt
`option(BUILD_CAMBRICON "" OFF)`
   
```txt
# TODO: add a cmake option
add_definitions(-DWITH_FAKE_DEVICE)
```
   
3. cmake/oneflow.cmake
```.cmake
if (BUILD_CAMBRICON)
  target_link_libraries(of_ccobj cnrt cnnl cndrv)
endif()
```

4. cmake/third_party.cmake 
```.cmake
if (BUILD_CAMBRICON)
  add_definitions(-DWITH_CAMBRICON)
  if (NOT DEFINED ENV{NEUWARE_HOME})
    message(FATAL_ERROR "Environment variable NEUWARE_HOME NOT found")
  endif()
  include_directories("$ENV{NEUWARE_HOME}/include")
  link_directories("$ENV{NEUWARE_HOME}/lib64")
endif()
```

5. docker/package/manylinux/build_wheel.py
```.py
def get_common_docker_args(
    oneflow_src_dir=None,
    cache_dir=None,
    current_dir=None,
    house_dir=None,
    use_system_proxy=True,
    cambricon=None, ## 新添加的 cambricon
):
    root = Path(cache_dir)
    child = Path(current_dir)
    assert root in child.parents
    cwd = os.getcwd()
    pwd_arg = f"-v {cwd}:{cwd}"
    cache_dir_arg = f"-v {cache_dir}:{cache_dir}"
    house_dir_arg = ""
    if house_dir:
        house_dir_arg = f"-v {house_dir}:{house_dir}"
    build_dir_arg = get_build_dir_arg(cache_dir, oneflow_src_dir)
    proxy_env_arg = get_proxy_env_args() if use_system_proxy else ""
    cambricon_arg = "--env NEUWARE_HOME=/usr/local/neuware" if cambricon else "" ## 新添加的 cambricon
    return f"-v {oneflow_src_dir}:{oneflow_src_dir} {proxy_env_arg} {pwd_arg} {cambricon_arg} {house_dir_arg} {cache_dir_arg} {build_dir_arg} -w {current_dir} --shm-size=8g"  ## 新添加的 cambricon
```
[修改细节多，详见](https://github.com/Oneflow-Inc/oneflow_cambricon/pull/17/files#diff-f421848746ad2a399bbe49e938fd01f333b9b1202a5cbca55fb5ead90d5571fd)

6. oneflow/api/python/env/env.h
```.h
#ifdef WITH_CAMBRICON
#include "cnrt.h"
#endif
```

7. oneflow/core/actor/actor.cpp
```.cpp
#include "oneflow/core/device/cambricon_device_context.h"
#include "oneflow/core/device/fake_device_device_context.h"
```

8. oneflow/core/actor/copy_hd_actor.cpp
```.cpp
#if defined(WITH_CUDA) or defined(WITH_FAKE_DEVICE) or defined(WITH_CAMBRICON)
```

9. 在 device_type.proto 中 enum DeviceType 添加新的设备类型

oneflow/core/common/device_type.proto
```.proto
enum DeviceType {
  kInvalidDevice = 0;
  kCPU = 1;
  kGPU = 2;
  kFAKEDEVICE = 3;
  kCambricon = 4;
}
```

10.  oneflow/core/common/id_util.h
```.h
class MemZoneId {
 public:
  using device_index_t = uint32_t;

  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static size_t kMaxDeviceTypeVal = (size_t{1} << kDeviceTypeBits) - size_t{1};
  constexpr static device_index_t kMaxDeviceIndex =
      (device_index_t{1} << kDeviceIndexBits) - device_index_t{1};

  MemZoneId() {
    device_type_ = DeviceType::kCPU;
    device_index_ = 0;
  }
  MemZoneId(DeviceType device_type, device_index_t device_index)
      : device_type_(device_type), device_index_(device_index) {
    CHECK_LE(static_cast<size_t>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }
  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }
  bool operator==(const MemZoneId& rhs) const {
    return device_type_ == rhs.device_type_ && device_index_ == rhs.device_index_;
  }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = std::hash<size_t>{}(static_cast<size_t>(device_type_));
    HashCombine(&hash, std::hash<device_index_t>{}(device_index_));
    return hash;
  }

 private:
  DeviceType device_type_;
  device_index_t device_index_;
};
```

```.h
template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const { return mem_zone_id.hash(); }
};

```

11. oneflow/core/device/cambricon_device_context.h
```.h
#ifndef ONEFLOW_CORE_DEVICE_CAMBRICON_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CAMBRICON_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cambricon_queue_handle.h"

namespace oneflow {

#ifdef WITH_CAMBRICON

class CambriconDeviceCtx : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CambriconDeviceCtx);
  CambriconDeviceCtx() = delete;
  ~CambriconDeviceCtx() override = default;

  explicit CambriconDeviceCtx(CambriconQueueHandle* queue_handler)
      : cambricon_handler_(queue_handler) {}

  const cnrtQueue_t& cambricon_queue() const override {
    return *(cambricon_handler_->cambricon_queue());
  }

  const cnnlHandle_t& cambricon_handle() const override {
    return *(cambricon_handler_->cambricon_handle());
  }

  void SyncDevice() override { CNRT_CHECK(cnrtSyncQueue(cambricon_queue())); }

  void AddCallBack(std::function<void()> callback) const override {
    cambricon_handler_->AddCallBack(callback);
  }

 protected:
  CambriconQueueHandle* cambricon_handler_;
};

#endif  // WITH_CAMBRICON

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CAMBRICON_DEVICE_CONTEXT_H_
```
12. oneflow/core/device/cambricon_device_context.cpp
```.cpp
#include "oneflow/core/device/cambricon_device_context.h"
#include "oneflow/core/thread/thread_context.h"

namespace oneflow {

#ifdef WITH_CAMBRICON

REGISTER_DEVICE_CONTEXT(DeviceType::kCambricon, ([](const ThreadCtx& thread_ctx) -> DeviceCtx* {
                          CambriconQueueHandle* cambricon_queue = nullptr;
                          cambricon_queue = thread_ctx.g_cambricon_queue.get();
                          return new CambriconDeviceCtx(cambricon_queue);
                        }));

#endif  // WITH_CAMBRICON

}  // namespace oneflow
```

13. oneflow/core/device/cambricon_device_stream_index.h
```.h
#ifndef ONEFLOW_CORE_DEVICE_CAMBRICON_DEVICE_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_CAMBRICON_DEVICE_STREAM_INDEX_H_

#ifdef WITH_CAMBRICON
#include "oneflow/core/device/stream_index.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CAMBRICONDeviceStreamIndexGenerator final : public StreamIndexGenerator {
 public:
  CAMBRICONDeviceStreamIndexGenerator() = default;
  OF_DISALLOW_COPY_AND_MOVE(CAMBRICONDeviceStreamIndexGenerator);
  ~CAMBRICONDeviceStreamIndexGenerator() = default;

  stream_index_t GenerateComputeStreamIndex() override { return 0; }
  stream_index_t GenerateH2DStreamIndex() override { return 1; }
  stream_index_t GenerateD2HStreamIndex() override { return 2; }
};

}  // namespace oneflow

#endif  // WITH_CAMBRICON

#endif  // ONEFLOW_CORE_DEVICE_CAMBRICON_DEVICE_STREAM_INDEX_H_
```
14. oneflow/core/device/cambricon_device_stream_index.cpp
```.cpp
#ifdef WITH_CAMBRICON
#include "oneflow/core/device/cambricon_device_stream_index.h"

namespace oneflow {
REGISTER_STREAM_INDEX_GENERATOR(DeviceType::kCambricon, CAMBRICONDeviceStreamIndexGenerator);
}
#endif  // WITH_CAMBRICON
```

15. oneflow/core/device/cambricon_queue_handle.h 
```.h
#ifndef ONEFLOW_CORE_DEVICE_CAMBRICON_QUEUE_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_CAMBRICON_QUEUE_HANDLE_H_

#ifdef WITH_CAMBRICON

#include "oneflow/core/common/channel.h"
#include "oneflow/core/device/cuda_util.h"
#include "cnrt.h"
#include "cnnl.h"

namespace oneflow {

struct CambriconCBNotifier {
  std::function<void()> callback;
  cnrtNotifier_t notifier;
};

class CambriconQueueHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CambriconQueueHandle);
  CambriconQueueHandle() = delete;
  CambriconQueueHandle(Channel<CambriconCBNotifier>* cb_notifier_chan)
      : cb_notifier_chan_(cb_notifier_chan) {}

  const cnrtQueue_t* cambricon_queue();
  const cnnlHandle_t* cambricon_handle();

  void AddCallBack(std::function<void()> callback);

  ~CambriconQueueHandle();

 private:
  Channel<CambriconCBNotifier>* cb_notifier_chan_;
  std::unique_ptr<cnrtQueue_t> cambricon_queue_;
  std::unique_ptr<cnnlHandle_t> cambricon_handle_;
};

}  // namespace oneflow
#endif  // WITH_CAMBRICON

#endif  // ONEFLOW_CORE_DEVICE_CAMBRICON_QUEUE_HANDLE_H_
```
16. oneflow/core/device/cambricon_queue_handle.cpp
```.cpp
#ifdef WITH_CAMBRICON
#include "oneflow/core/device/cambricon_queue_handle.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

const cnrtQueue_t* CambriconQueueHandle::cambricon_queue() {
  if (!cambricon_queue_) {
    cambricon_queue_.reset(new cnrtQueue_t);
    CNRT_CHECK(cnrtCreateQueue(cambricon_queue_.get()));
  }
  return cambricon_queue_.get();
}

const cnnlHandle_t* CambriconQueueHandle::cambricon_handle() {
  if (!cambricon_handle_) {
    cambricon_handle_.reset(new cnnlHandle_t);
    CNNL_CHECK(cnnlCreate(cambricon_handle_.get()));
    CNNL_CHECK(cnnlSetQueue(*cambricon_handle_.get(), *cambricon_queue()));
  }
  return cambricon_handle_.get();
}

CambriconQueueHandle::~CambriconQueueHandle() {
  if (cambricon_queue_) { CNRT_CHECK(cnrtDestroyQueue(*cambricon_queue_)); }
  if (cambricon_handle_) { CNNL_CHECK(cnnlDestroy(*cambricon_handle_)); }
}

void CambriconQueueHandle::AddCallBack(std::function<void()> callback) {
  CambriconCBNotifier cb_notifier;
  cb_notifier.callback = std::move(callback);
  CNRT_CHECK(cnrtCreateNotifier(&(cb_notifier.notifier)));
  CNRT_CHECK(cnrtPlaceNotifier(cb_notifier.notifier, *cambricon_queue()));
  cb_notifier_chan_->Send(cb_notifier);
}

}  // namespace oneflow
#endif  // WITH_CAMBRICON
```

17. 在 device_context.h 里增加如下两个虚函数

oneflow/core/device/device_context.h
```.h
#ifdef WITH_CAMBRICON
#include "cnrt.h"
#include "cnnl.h"
#endif  // WITH_CAMBRICON

#ifdef WITH_CAMBRICON
  virtual const cnnlHandle_t& cambricon_handle() const { UNIMPLEMENTED(); }
  virtual const cnrtQueue_t& cambricon_queue() const { UNIMPLEMENTED(); }
#endif  // WITH_CAMBRICON
```


18. 添加 fake_device_device_context 头文件和源文件。主要是定义和注册一个 FakeDeviceDeviceCtx 类<br>

oneflow/core/device/fake_device_device_context.h
```.h
#ifndef ONEFLOW_CORE_DEVICE_FAKE_DEVICE_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_FAKE_DEVICE_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class FakeDeviceDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FakeDeviceDeviceCtx);
  FakeDeviceDeviceCtx() = default;
  ~FakeDeviceDeviceCtx() = default;

  std::unique_ptr<DeviceCtx> Copy() const {
    return std::unique_ptr<DeviceCtx>(new FakeDeviceDeviceCtx());
  }

  void SyncDevice() override {}

  void AddCallBack(std::function<void()> callback) const override { callback(); }

};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_FAKE_DEVICE_DEVICE_CONTEXT_H_
```

19. oneflow/core/device/fake_device_device_context.cpp
```.cpp
#include "oneflow/core/device/fake_device_device_context.h"
#include "oneflow/core/thread/thread_context.h"

namespace oneflow {

#ifdef WITH_FAKE_DEVICE

REGISTER_DEVICE_CONTEXT(DeviceType::kFAKEDEVICE, ([](const ThreadCtx& thread_ctx) -> DeviceCtx* {
                          return new FakeDeviceDeviceCtx();
                        }));

#endif  // WITH_FAKE_DEVICE

}  // namespace oneflow
```

20. 添加 fake_device_stream_index 头文件和源文件: 主要是定义和注册一个 FakeDeviceStreamIndexGenerator 类

oneflow/core/device/fake_device_stream_index.h
```.h
#ifndef ONEFLOW_CORE_DEVICE_FAKE_DEVICE_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_FAKE_DEVICE_STREAM_INDEX_H_

#include "oneflow/core/device/stream_index.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class FakeDeviceStreamIndexGenerator final : public StreamIndexGenerator {
 public:
  FakeDeviceStreamIndexGenerator() = default;
  OF_DISALLOW_COPY_AND_MOVE(FakeDeviceStreamIndexGenerator);
  ~FakeDeviceStreamIndexGenerator() = default;

  stream_index_t GenerateComputeStreamIndex() override { return 0; }
  stream_index_t GenerateH2DStreamIndex() override { return 1; }
  stream_index_t GenerateD2HStreamIndex() override { return 2; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_FAKE_DEVICE_STREAM_INDEX_H_
```
21. oneflow/core/device/fake_device_stream_index.cpp
```.cpp
#include "oneflow/core/device/fake_device_stream_index.h"

namespace oneflow {
REGISTER_STREAM_INDEX_GENERATOR(DeviceType::kFAKEDEVICE, FakeDeviceStreamIndexGenerator);
}
```

22. oneflow/core/device/mlu_util.h
```.h
#ifndef ONEFLOW_CORE_DEVICE_MLU_UTIL_H_
#define ONEFLOW_CORE_DEVICE_MLU_UTIL_H_

#ifdef WITH_CAMBRICON

#include <cnrt.h>
#include <cnnl.h>

namespace oneflow {

class MLUCurrentDeviceGuard final {
 public:
  MLUCurrentDeviceGuard(int ordinal) {
    CNRT_CHECK(cnrtGetCurrentDevice(&saved_dev_));
    cnrtDev_t cur_dev;
    CNRT_CHECK(cnrtGetDeviceHandle(&cur_dev, ordinal));
    CNRT_CHECK(cnrtSetCurrentDevice(cur_dev));
  }
  MLUCurrentDeviceGuard() { CNRT_CHECK(cnrtGetCurrentDevice(&saved_dev_)); }
  ~MLUCurrentDeviceGuard() { CNRT_CHECK(cnrtSetCurrentDevice(saved_dev_)); }
  MLUCurrentDeviceGuard(const MLUCurrentDeviceGuard&) = delete;
  MLUCurrentDeviceGuard& operator=(const MLUCurrentDeviceGuard&) = delete;
  MLUCurrentDeviceGuard(MLUCurrentDeviceGuard&&) = delete;
  MLUCurrentDeviceGuard& operator=(MLUCurrentDeviceGuard&&) = delete;

 private:
  cnrtDev_t saved_dev_;
};

}  // namespace oneflow

#endif  // WITH_CAMBRICON

#endif  // ONEFLOW_CORE_DEVICE_MLU_UTIL_H_
```

23. oneflow/core/framework/device_register_cambricon.h
```.h
#ifndef ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTER_CAMBRICON_H_
#define ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTER_CAMBRICON_H_
#ifdef WITH_CAMBRICON
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {

REGISTER_DEVICE(DeviceType::kCambricon)
    .SetDumpVersionInfoFn([]() -> void {})
    .SetDeviceTag("cambricon");
}  // namespace oneflow
#endif  // WITH_CAMBRICON
#endif  // ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTER_CAMBRICON_H_
```

24. 在 oneflow/core/framework 文件夹下添加 device_register_fakedev.h

oneflow/core/framework/device_register_fakedev.h 
```.h
#ifndef ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTER_FAKEDEV_H_
#define ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTER_FAKEDEV_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {
constexpr int FAKE_MAGIC_CODE = 0x46414B45;

struct fakeFloat16 {
  unsigned short int value_;
};

template<typename T>
struct IsFloat16;

template<>
struct IsFloat16<fakeFloat16> : std::true_type {};

REGISTER_DEVICE(DeviceType::kFAKEDEVICE)
    .SetDumpVersionInfoFn([]() -> void {})
    .SetDeviceTag("fakedevice");
}  // namespace oneflow
#endif  // ONEFLOW_CORE_FRAMEWORK_DEVICE_REGISTER_FAKEDEV_H_
```

25. oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.cpp
将 `Global<IDMgr>::Get()->CpuMemZoneId()` 替换为 `MemZoneId(DeviceType::kCPU, 0)`

26. oneflow/core/graph/boxing/sub_task_graph_builder_context.h
```.h
#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_

#include "oneflow/core/common/id_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

class TaskGraph;
class TaskNode;

class SubTskGphBuilderCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SubTskGphBuilderCtx);
  explicit SubTskGphBuilderCtx(TaskGraph* task_graph);
  virtual ~SubTskGphBuilderCtx() = default;

  virtual TaskGraph* task_graph();
  TaskNode* GetProxyNode(TaskNode* src_node, MemZoneId src_mem_zone_id, int64_t dst_machine_id,
                         MemZoneId dst_mem_zone_id);
  TaskNode* GetProxyNode(TaskNode* src_node, MemZoneId src_mem_zone_id,
                         const ParallelDesc& dst_parallel_desc, const int64_t dst_parallel_id);
  template<typename T1, typename T2>
  void ConnectAll121(const std::vector<T1*>& src_nodes, const std::vector<T2*>& dst_nodes) {
    CHECK_EQ(src_nodes.size(), dst_nodes.size());
    FOR_RANGE(int64_t, i, 0, dst_nodes.size()) {
      Connect<TaskNode>(src_nodes.at(i), task_graph()->NewEdge(), dst_nodes.at(i));
    }
  }

 private:
  TaskGraph* task_graph_;
  HashMap<TaskNode*, HashMap<std::pair<int64_t, MemZoneId>, TaskNode*>> node2proxies_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
```

27. oneflow/core/graph/boxing/sub_task_graph_builder_context.cpp
```.cpp
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"

namespace oneflow {

SubTskGphBuilderCtx::SubTskGphBuilderCtx(TaskGraph* task_graph) : task_graph_(task_graph) {}

TaskGraph* SubTskGphBuilderCtx::task_graph() { return task_graph_; }

TaskNode* SubTskGphBuilderCtx::GetProxyNode(TaskNode* src_node, MemZoneId src_mem_zone_id,
                                            int64_t dst_machine_id, MemZoneId dst_mem_zone_id) {
  const auto key = std::make_pair(dst_machine_id, dst_mem_zone_id);
  if (node2proxies_.find(src_node) != node2proxies_.cend()
      && node2proxies_.at(src_node).find(key) != node2proxies_.at(src_node).cend()) {
    return node2proxies_.at(src_node).at(key);
  } else {
    if (dst_machine_id == src_node->machine_id() && dst_mem_zone_id == src_mem_zone_id) {
      node2proxies_[src_node][key] = src_node;
      return src_node;
    } else if (dst_mem_zone_id.device_type() == DeviceType::kGPU) {
      TaskNode* proxy_on_dst_host =
          GetProxyNode(src_node, src_mem_zone_id, dst_machine_id, MemZoneId(DeviceType::kCPU, 0));
      CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, proxy_on_dst_host->machine_id(),
                      dst_mem_zone_id.device_type(), dst_mem_zone_id.device_index());
      Connect<TaskNode>(proxy_on_dst_host, task_graph()->NewEdge(), copy_task);
      node2proxies_[src_node][key] = copy_task;
      return copy_task;
    } else if (dst_mem_zone_id.device_type() == DeviceType::kCPU) {
      if (src_node->machine_id() == dst_machine_id) {
        if (src_mem_zone_id.device_type() == DeviceType::kGPU) {
          CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
          copy_task->Init(CopyHdOpConf::D2H, src_node->machine_id(), src_mem_zone_id.device_type(),
                          src_mem_zone_id.device_index());
          Connect<TaskNode>(src_node, task_graph()->NewEdge(), copy_task);
          node2proxies_[src_node][key] = copy_task;
          return copy_task;
        } else {
          UNIMPLEMENTED();
        }
      } else {
        TaskNode* proxy_on_src_host = GetProxyNode(
            src_node, src_mem_zone_id, src_node->machine_id(), MemZoneId(DeviceType::kCPU, 0));
        CopyCommNetTaskNode* copy_comm_net_task = task_graph()->NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task->Init(dst_machine_id);
        Connect<TaskNode>(proxy_on_src_host, task_graph()->NewEdge(), copy_comm_net_task);
        node2proxies_[src_node][key] = copy_comm_net_task;
        return copy_comm_net_task;
      }
    } else {
      UNIMPLEMENTED();
    }
  }
}

TaskNode* SubTskGphBuilderCtx::GetProxyNode(TaskNode* src_node, const MemZoneId src_mem_zone_id,
                                            const ParallelDesc& dst_parallel_desc,
                                            const int64_t dst_parallel_id) {
  const int64_t dst_machine_id =
      CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(dst_parallel_id));
  MemZoneId dst_mem_zone_id;
  const IDMgr* id_mgr = Global<IDMgr>::Get();
  if (dst_parallel_desc.device_type() == DeviceType::kCPU) {
    dst_mem_zone_id = MemZoneId(DeviceType::kCPU, 0);
  } else if (dst_parallel_desc.device_type() == DeviceType::kGPU) {
    const int64_t dst_dev_phy_id =
        CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(dst_parallel_id));
    dst_mem_zone_id = MemZoneId(DeviceType::kGPU, dst_dev_phy_id);
  } else {
    UNIMPLEMENTED();
  }
  return GetProxyNode(src_node, src_mem_zone_id, dst_machine_id, dst_mem_zone_id);
}

}  // namespace oneflow
```

28. oneflow/core/graph/copy_task_node.h 
```.h
#ifndef ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CopyTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyTaskNode);
  CopyTaskNode() = default;
  virtual ~CopyTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

 protected:
  virtual OperatorConf NewCopyOpConf() = 0;

 private:
  void InferProducedDataRegstTimeShape() final;
};

class CopyHdTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdTaskNode);
  CopyHdTaskNode() = default;
  ~CopyHdTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyHd; }

  void Init(CopyHdOpConf::Type, int64_t machine_id, DeviceType dev_type, int64_t dev_phy_id);

  CopyHdOpConf::Type copy_type() const { return copy_type_; }
  MemZoneId MemZoneId121() const override {
    if (copy_type_ == CopyHdOpConf::H2D) {
      return TaskNode::MemZoneId121();
    } else if (copy_type_ == CopyHdOpConf::D2H) {
      return MemZoneId(DeviceType::kCPU, 0);
    } else {
      UNIMPLEMENTED();
    }
    return MemZoneId(DeviceType::kCPU, 0);
  }

 private:
  void InitProducedRegstMemCase(MemoryCase*) override;
  OperatorConf NewCopyOpConf() override;

  CopyHdOpConf::Type copy_type_;
};

class CopyCommNetTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetTaskNode);
  CopyCommNetTaskNode() = default;
  ~CopyCommNetTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyCommNet; }

  void Init(int64_t machine_id);

 private:
  void InitProducedRegstMemCase(MemoryCase*) override;
  void PinConsumedRegstMemCase(MemoryCase*) override;
  OperatorConf NewCopyOpConf() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
```

29. oneflow/core/graph/copy_task_node.cpp
```.cpp
/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/device/cpu_stream_index.h"

namespace oneflow {

void CopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::string name("copy_out");
  std::shared_ptr<RegstDesc> out_regst(nullptr);
  CopyHdTaskNode* copy_hd = dynamic_cast<CopyHdTaskNode*>(this);
  if (copy_hd != nullptr) {
    TaskNode* first_dst_node = nullptr;
    ForEachNodeOnOutDataEdge([&](TaskNode* node) {
      if (first_dst_node == nullptr) { first_dst_node = node; }
    });
    if (out_regst == nullptr) {
      // normal copy hd task can reuse mem
      out_regst = ProduceRegst(name, true);
    }
  }
  if (out_regst == nullptr) {
    // copy comm_net task cannot reuse mem
    out_regst = ProduceRegst(name, false);
  }
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst(name, out_regst); });
}

void CopyTaskNode::ConsumeAllRegsts() { ConsumeRegst("copy_in", SoleInDataEdge()->GetSoleRegst()); }

void CopyTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetSoleConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp(NewCopyOpConf());
  node->BindBnWithRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnWithRegst(node->op()->SoleObn(), out_regst);
}

void CopyTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

void CopyHdTaskNode::Init(CopyHdOpConf::Type copy_type, int64_t machine_id, DeviceType dev_type,
                          int64_t dev_phy_id) {
  copy_type_ = copy_type;
  set_machine_id(machine_id);
  DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), dev_type,
                     static_cast<DeviceId::device_index_t>(dev_phy_id)};
  auto* stream_index_generator =
      Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id);
  StreamId::stream_index_t stream_index = 0;
  if (copy_type == CopyHdOpConf::H2D) {
    stream_index = stream_index_generator->GenerateH2DStreamIndex();
  } else if (copy_type == CopyHdOpConf::D2H) {
    stream_index = stream_index_generator->GenerateD2HStreamIndex();
  } else {
    UNIMPLEMENTED();
  }
  set_thrd_id(SerializeStreamIdToInt64(StreamId{device_id, stream_index}));
}

void CopyHdTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (copy_type_ == CopyHdOpConf::H2D) {
    TaskNode::InitProducedRegstMemCase(mem_case);
  } else if (copy_type_ == CopyHdOpConf::D2H) {
    DeviceType dev_type = DeserializeStreamIdFromInt64(thrd_id()).device_id().device_type();
    if (dev_type == DeviceType::kGPU) {
      mem_case->mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(GpuPhyId());
    } else if (dev_type == DeviceType::kFAKEDEVICE) {
      mem_case->mutable_host_mem();
    } else if (dev_type == DeviceType::kCambricon) {
      mem_case->mutable_host_mem();
    }
  } else {
    UNIMPLEMENTED();
  }
}

OperatorConf CopyHdTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_hd_" + NewUniqueId());
  conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type())));
  conf.mutable_copy_hd_conf()->set_type(copy_type_);
  auto in_regst = GetSoleConsumedRegst("copy_in");
  if (in_regst->NumOfLbi() == 1) {
    in_regst->ForEachLbi(
        [&](const LogicalBlobId& lbi) { *conf.mutable_copy_hd_conf()->mutable_lbi() = lbi; });
  }
  return conf;
}

void CopyCommNetTaskNode::Init(int64_t machine_id) {
  set_machine_id(machine_id);
  DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), DeviceType::kCPU,
                     DeviceId::kCPUDeviceIndex};
  auto* generator = dynamic_cast<CPUStreamIndexGenerator*>(
      Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
  CHECK_NOTNULL(generator);
  StreamId stream_id{device_id, generator->GenerateCommNetStreamIndex()};
  set_thrd_id(SerializeStreamIdToInt64(stream_id));
}

void CopyCommNetTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  mem_case->mutable_host_mem()->set_used_by_network(true);
}

void CopyCommNetTaskNode::PinConsumedRegstMemCase(MemoryCase* mem_case) {
  CHECK(mem_case->has_host_mem());
  mem_case->mutable_host_mem()->set_used_by_network(true);
}

OperatorConf CopyCommNetTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_comm_net_" + NewUniqueId());
  conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(this->device_type())));
  conf.mutable_copy_comm_net_conf();
  return conf;
}

}  // namespace oneflow

```

30. oneflow/core/graph/exec_graph.cpp
```.cpp
#include "oneflow/core/graph/exec_graph.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void ExecNode::BindBnWithRegst(const std::string& bn, std::shared_ptr<RegstDesc> regst) {
  CHECK(bn_in_op2regst_.emplace(bn, regst).second);
}

void ExecNode::BindBnsWithRegst(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                                std::shared_ptr<RegstDesc> regst) {
  for (const std::string& bn : (op_.get()->*bns_getter)()) { BindBnWithRegst(bn, regst); }
}

void ExecNode::AddBnToRegstAndBindIt(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                                     std::shared_ptr<RegstDesc> regst) {
  for (const std::string& bn : (op_.get()->*bns_getter)()) { regst->AddLbi(op_->BnInOp2Lbi(bn)); }
  BindBnsWithRegst(bns_getter, regst);
}

bool ExecNode::TryBindBnWithOneOfTheRegsts(const std::string& bn,
                                           const std::list<std::shared_ptr<RegstDesc>>& regsts) {
  const LogicalBlobId& lbi = op()->BnInOp2Lbi(bn);
  bool has_binded = false;
  LOG(INFO) << "start";
  for (std::shared_ptr<RegstDesc> regst : regsts) {
    LOG(INFO) << lbi.DebugString();
    LOG(INFO) << "lbi in regst:";
    regst->ForEachLbi([&regst](const LogicalBlobId& lbi) {
      LOG(INFO) << "lbi: " << lbi.DebugString();
      LOG(INFO) << regst->GetBlobDesc(lbi);
    });
    if (regst->GetBlobDesc(lbi) == nullptr) { continue; }
    LOG(INFO) << "xxxxxx";
    BindBnWithRegst(bn, regst);
    has_binded = true;
    break;
  }
  LOG(INFO) << "end";
  return has_binded;
}

void ExecNode::BindBnWithOneOfTheRegsts(const std::string& bn,
                                        const std::list<std::shared_ptr<RegstDesc>>& regsts) {
  CHECK(TryBindBnWithOneOfTheRegsts(bn, regsts));
}

void ExecNode::UnbindBnWithEmptyRegst() {
  EraseIf<std::string, std::shared_ptr<RegstDesc>>(
      &bn_in_op2regst_, [](HashMap<std::string, std::shared_ptr<RegstDesc>>::iterator it) {
        return it->second->regst_desc_type().has_data_regst_desc() && it->second->NumOfLbi() == 0;
      });
}

void ExecNode::ToProto(const ParallelContext* parallel_ctx, ExecNodeProto* ret) const {
  op_->GenKernelConf(GetBlobDesc4BnInOpFunc(), parallel_ctx, ret->mutable_kernel_conf());
  for (const auto& bn_regst : bn_in_op2regst_) {
    const std::string& bn_in_op = bn_regst.first;
    auto regst = bn_regst.second;
    CHECK(regst);
    PbMapPair<std::string, int64_t> pair{bn_in_op, regst->regst_desc_id()};
    CHECK(ret->mutable_bn_in_op2regst_desc_id()->insert(pair).second);
  }
}

namespace {

Maybe<void> CheckPhysicalBlobDesc(const BlobDesc& logical,
                                  const ParallelDistribution& parallel_distribution,
                                  const ParallelDesc& parallel_desc,
                                  const ParallelContext* parallel_ctx, const BlobDesc& physical) {
  CHECK_EQ_OR_RETURN(physical.shape(),
                     *CHECK_JUST(GetPhysicalShape(logical.shape(), parallel_distribution,
                                                  parallel_desc, *parallel_ctx)));
  return Maybe<void>::Ok();
}

Maybe<void> CheckPhysicalBlobDesc(
    const Operator& op, const PbRpf<std::string>& bns,
    const std::function<Maybe<const BlobDesc>(const std::string&)>& GetLogicalBlobDesc,
    const ParallelDistributionSignature* parallel_distribution_signature,
    const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& GetPhysicalBlobDesc) {
  const std::shared_ptr<const ParallelDesc> op_parallel_desc = CHECK_JUST(op.GetOpParallelDesc());
  for (const auto& bn : bns) {
    const BlobDesc* physical_blob_desc = GetPhysicalBlobDesc(bn);
    if (physical_blob_desc == nullptr) {
      // TODO(liujuncheng): remove this hotfix
      continue;
    }
    if (*CHECK_JUST(op.GetParallelDesc4BnInOp(bn)) == *op_parallel_desc) {
      CHECK_JUST(CheckPhysicalBlobDesc(
          *CHECK_JUST(GetLogicalBlobDesc(bn)),
          parallel_distribution_signature->bn_in_op2parallel_distribution().at(bn),
          *op_parallel_desc, parallel_ctx, *physical_blob_desc));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

void ExecNode::InferBlobDescs(const ParallelContext* parallel_ctx) {
  auto GetBlobDesc4BnInOp = GetBlobDesc4BnInOpFunc();
  const OpNode* op_node = Global<OpGraph>::Get()->OpNode4OpName(op()->op_name());
  const ParallelDistributionSignature* parallel_distribution_signature = nullptr;
  if (op_node != nullptr) {
    parallel_distribution_signature = &op_node->parallel_distribution_signature();
  }

  if (op_node != nullptr && parallel_ctx->parallel_num() > 1
      && parallel_distribution_signature != nullptr) {
    CheckPhysicalBlobDesc(
        *op(), op()->input_bns(),
        std::bind(&Operator::GetLogicalBlobDesc4Ibn, op().get(), std::placeholders::_1),
        parallel_distribution_signature, parallel_ctx, GetBlobDesc4BnInOp);
  }
  CHECK_JUST(op_->InferBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, &GlobalJobDesc()));
  if (op_node != nullptr && parallel_ctx->parallel_num() > 1
      && parallel_distribution_signature != nullptr) {
    CheckPhysicalBlobDesc(
        *op(), op()->output_bns(),
        std::bind(&Operator::GetLogicalBlobDesc4Obn, op().get(), std::placeholders::_1),
        parallel_distribution_signature, parallel_ctx, GetBlobDesc4BnInOp);
  }
  CHECK_JUST(op_->InferInplaceObn2IbnIf(&mut_inplace_obn2ibn_, &con_inplace_obn2ibn_,
                                        GetBlobDesc4BnInOp, parallel_ctx));
}

std::function<BlobDesc*(const std::string&)> ExecNode::GetBlobDesc4BnInOpFunc() const {
  return [this](const std::string& bn_in_op) -> BlobDesc* {
    auto it = bn_in_op2regst_.find(bn_in_op);
    if (it == bn_in_op2regst_.end()) { return nullptr; }
    std::shared_ptr<RegstDesc> regst = it->second;
    CHECK(regst);
    return regst->MutBlobDesc(op()->BnInOp2Lbi(bn_in_op));
  };
}

void ExecGraph::ToExecSequence(const ParallelContext* parallel_ctx, ExecSequence* ret) const {
  TopoForEachNode([&](ExecNode* node) { node->ToProto(parallel_ctx, ret->add_exec_node()); });
}

}  // namespace oneflow
```

31. oneflow/core/graph/id_serialization.h
```.h
int64_t SerializeMemZoneIdToInt64(const MemZoneId&);
MemZoneId DeserializeMemZoneIdFromInt64(int64_t);
```

32. oneflow/core/graph/id_serialization.cpp
```.cpp
constexpr size_t kMemZoneIdDeviceTypeShift = MemZoneId::kDeviceIndexBits;
constexpr int64_t kMemZoneIdDeviceTypeInt64Mask = ((int64_t{1} << MemZoneId::kDeviceTypeBits) - 1)
                                                  << kMemZoneIdDeviceTypeShift;
constexpr int64_t kMemZoneIdDeviceIndexInt64Mask = (int64_t{1} << MemZoneId::kDeviceIndexBits) - 1;

int64_t SerializeMemZoneIdToInt64(const MemZoneId& mem_zone_id) {
  int64_t id = static_cast<int64_t>(mem_zone_id.device_index());
  id |= static_cast<int64_t>(mem_zone_id.device_type())
        << stream_id_const::kMemZoneIdDeviceTypeShift;
  return id;
}

MemZoneId DeserializeMemZoneIdFromInt64(int64_t mem_zone_id) {
  int64_t device_type = (mem_zone_id & stream_id_const::kMemZoneIdDeviceTypeInt64Mask)
                        >> stream_id_const::kDeviceTypeShift;
  int64_t device_index = mem_zone_id & stream_id_const::kMemZoneIdDeviceIndexInt64Mask;
  return MemZoneId(static_cast<DeviceType>(device_type),
                   static_cast<MemZoneId::device_index_t>(device_index));
}
```

33. oneflow/core/graph/logical_node.h 
```.h
#include "oneflow/core/common/id_util.h"

using MutBufTaskFn = std::function<TaskNode**(CompTaskNode*, int64_t, MemZoneId)>;

#define BLD_SUB_TSK_GPH_MTHD_ARGS()                                \
  (const LogicalNode* src_logical, const LogicalNode* dst_logical, \
   const std::vector<CompTaskNode*>& sorted_src_comp_tasks,        \
   const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, MutBufTaskFn MutBufTask)
```

34. oneflow/core/graph/slice_boxing_task_node.h
将 `int64_t mem_zone_id` 替换为 `MemZoneId mem_zone_id`

35. oneflow/core/graph/slice_boxing_task_node.cpp
```.cpp
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"

namespace oneflow {

void SliceBoxingTaskNode::Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice,
                               const SliceBoxingTaskMode mode, int64_t machine_id, int64_t thrd_id,
                               MemZoneId mem_zone_id) {
  lbi_ = lbi;
  out_slice_ = out_slice;
  out_shape_ = out_slice.shape();
  mode_ = mode;
  mem_zone_id_ = mem_zone_id;
  set_machine_id(machine_id);
  set_thrd_id(thrd_id);
}

void SliceBoxingTaskNode::Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice,
                               const SliceBoxingTaskMode mode, int64_t machine_id,
                               int64_t thrd_id) {
  IDMgr* global_id_mgr = Global<IDMgr>::Get();
  DeviceType device_type = global_id_mgr->GetDeviceTypeFromThrdId(thrd_id);
  MemZoneId mem_zone_id;
  if (device_type == DeviceType::kCPU) {
    mem_zone_id = MemZoneId(DeviceType::kCPU, 0);
  } else if (device_type == DeviceType::kGPU) {
    mem_zone_id = MemZoneId(DeviceType::kGPU, global_id_mgr->GetGpuPhyIdFromThrdId(thrd_id));
  } else {
    UNIMPLEMENTED();
  }
  Init(lbi, out_slice, mode, machine_id, thrd_id, mem_zone_id);
}

void SliceBoxingTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst_desc = ProduceRegst("out", false, 2, 2);
  this->ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst_desc); });
  ProduceRegst("tmp", false, 1, 1);
}

void SliceBoxingTaskNode::ConsumeAllRegsts() {
  HashMap<const TaskEdge*, int64_t> edge2order_;
  FOR_RANGE(int64_t, i, 0, ordered_in_data_edges_.size()) {
    edge2order_.emplace(ordered_in_data_edges_.at(i), i);
  }
  int64_t in_data_edge_cnt = 0;
  ForEachInDataEdge([&](TaskEdge* edge) {
    const auto order_it = edge2order_.find(edge);
    CHECK(order_it != edge2order_.end());
    ConsumeRegst("in_" + std::to_string(order_it->second), edge->GetSoleRegst());
    in_data_edge_cnt += 1;
  });
  CHECK_EQ(in_data_edge_cnt, ordered_in_data_edges_.size());
}

void SliceBoxingTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> op = ConstructOp(GetBoxingOpConf());
  node->mut_op() = op;
  FOR_RANGE(size_t, i, 0, op->input_bns().size()) {
    const std::string& ibn = op->input_bns().Get(i);
    CHECK_EQ(GenUnRepeatedBn(ibn).second, i);
    node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(lbi_);
  node->BindBnWithRegst(op->SoleObn(), out_regst);
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(parallel_ctx());
}

void SliceBoxingTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void SliceBoxingTaskNode::SetInDataEdgeSlice(const TaskEdge* edge, const TensorSliceView& slice) {
  CHECK(in_data_edge2slice_.emplace(edge, slice).second);
  ordered_in_data_edges_.push_back(edge);
}

void SliceBoxingTaskNode::ConnectToSrcNodeWithSlice(TaskNode* src, TaskEdge* edge,
                                                    const TensorSliceView& slice) {
  Connect<TaskNode>(src, edge, this);
  SetInDataEdgeSlice(edge, slice);
}

void SliceBoxingTaskNode::SetOutShape(const Shape& shape) { out_shape_ = shape; }

OperatorConf SliceBoxingTaskNode::GetBoxingOpConf() {
  OperatorConf op_conf{};
  op_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(device_type())));
  SliceBoxingConf boxing_conf{};
  *boxing_conf.mutable_lbi() = lbi_;
  out_slice_.ToProto(boxing_conf.mutable_out_slice());
  out_shape_.ToProto(boxing_conf.mutable_out_shape());
  for (const TaskEdge* edge : ordered_in_data_edges_) {
    in_data_edge2slice_.at(edge).ToProto(boxing_conf.mutable_in_slice()->Add());
  }
  if (mode_ == kSliceBoxingTaskModeCopy) {
    op_conf.set_name("System-Boxing-BoxingCopy-" + NewUniqueId());
    SliceBoxingCopyOpConf* conf = op_conf.mutable_slice_boxing_copy_conf();
    *conf->mutable_slice_boxing_conf() = boxing_conf;
  } else if (mode_ == kSliceBoxingTaskModeAdd) {
    op_conf.set_name("System-Boxing-BoxingAdd-" + NewUniqueId());
    SliceBoxingAddOpConf* conf = op_conf.mutable_slice_boxing_add_conf();
    *conf->mutable_slice_boxing_conf() = boxing_conf;
  } else {
    UNIMPLEMENTED();
  }
  return op_conf;
}

void SliceBoxingTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (mem_zone_id_.device_type() == DeviceType::kCPU) {
    HostMemory* host_mem = mem_case->mutable_host_mem();
    if (device_type() == DeviceType::kGPU) {
      host_mem->mutable_cuda_pinned_mem()->set_device_id(GpuPhyId());
    }
  } else if (mem_zone_id_.device_type() == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(mem_zone_id_.device_index());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
```

36. oneflow/core/graph/task_graph.h
```.h
void BuildTaskPath(CompTaskNode* src, CompTaskNode* dst, MutBufTaskFn MutBufTask,
                     bool use_buf_task_node);
  using GetBufTaskFn = std::function<TaskNode*(int64_t, MemZoneId)>;
  using SetBufTaskFn = std::function<TaskNode*(int64_t, MemZoneId, TaskNode*)>;
  TaskNode* BuildTaskStep(TaskNode* src, TaskNode* dst, const GetBufTaskFn& GetBufTask,
                          const SetBufTaskFn& SetBufTask, bool use_buf_task_node);
```

37. oneflow/core/graph/task_graph.cpp
```.cpp
/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/graph/inplace_lbi_graph.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2p_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/b21_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/boxing_identity_task_node.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"

namespace oneflow {

namespace {

bool IsInterfaceTask(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  auto op_type_case = comp_task_node->logical_node()->SoleOp()->op_conf().op_type_case();
  return IsClassRegistered<int32_t, IsInterfaceOpConf4OpTypeCase>(op_type_case);
}

bool IsConnectToTickOp(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  const Operator* op = comp_task_node->logical_node()->SoleOp().get();
  if (dynamic_cast<const VariableOp*>(op) != nullptr) { return true; }
  return false;
}

bool IsOptimizerPassOp(const Operator* op) {
  // NOTE(chengcheng): use scope::calculation_pass_name instead of area_id to not merge optimizer
  // ops with fw/bw ops
  if (!op->op_conf().has_scope_symbol_id()) {
    // NOTE(chengcheng): Some system op insert to OpGraph may not set scope_symbol_id, it MUST NOT
    // optimizer subgraph ops.
    return false;
  }
  int64_t scope_symbol_id = op->op_conf().scope_symbol_id();
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id))
      << " Error! op : \n " << op->op_conf().DebugString()
      << " has error scope_symbol_id = " << scope_symbol_id
      << " which cannot find in Global<symbol::Storage<Scope>>::Get()\n";
  const Scope& scope = Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
  return scope.scope_proto().calculation_pass_name() == kOptimizerPass;
}

bool IsSpecialOpNotConsiderMergeInChain(const Operator* op) {
  const OperatorConf& op_conf = op->op_conf();
  if (op_conf.has_variable_conf() || op_conf.has_tick_conf() || op_conf.has_device_tick_conf()
      || op_conf.has_src_subset_tick_conf() || op_conf.has_dst_subset_tick_conf()
      || op_conf.has_source_tick_conf() || op_conf.has_sink_tick_conf()
      || op_conf.has_acc_tick_conf()) {
    return true;
  }
  // NOTE(chengcheng): ONLY nccl_use_compute_stream = false will exclude optimizer pass ops
  if (!Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
      && IsOptimizerPassOp(op)) {
    return true;
  }
  return false;
}

bool IsTaskNodeProducedResgtHasMultiRegstNum(const TaskNode* node) {
  for (const auto& pair : node->produced_regsts()) {
    if (pair.second->min_register_num() > 1) { return true; }
  }
  return false;
}

bool CanBeMergedInChain(const TaskNode* node) {
  // ONLY the node which is NormalForward and in GPU and NOT variable can be merged.
  if (IsTaskNodeProducedResgtHasMultiRegstNum(node)) { return false; }
  const auto* fw_comp_node = dynamic_cast<const NormalForwardCompTaskNode*>(node);
  if (fw_comp_node == nullptr) { return false; }
  if (fw_comp_node->logical_node()->op_vec().size() != 1) { return false; }
  if (fw_comp_node->device_type() != DeviceType::kGPU) { return false; }
  const Operator* op = fw_comp_node->logical_node()->SoleOp().get();
  if (IsSpecialOpNotConsiderMergeInChain(op)) { return false; }
  return true;
}

void TraverseConnectedSubGraphMergeInThisChain(TaskNode* this_node, const int64_t this_chain_id) {
  CHECK_NE(this_chain_id, -1);
  CHECK_EQ(this_node->chain_id(), -1);
  // bfs search all node can be merged in this chain
  HashSet<TaskNode*> visited_nodes;
  std::queue<TaskNode*> queued_nodes;
  queued_nodes.push(this_node);
  visited_nodes.insert(this_node);
  while (!queued_nodes.empty()) {
    TaskNode* cur_node = queued_nodes.front();
    queued_nodes.pop();

    CHECK_EQ(cur_node->chain_id(), -1);
    cur_node->set_chain_id(this_chain_id);

    cur_node->ForEachNodeOnInOutEdge([&](TaskNode* next_node) {
      if (visited_nodes.find(next_node) == visited_nodes.end() && CanBeMergedInChain(next_node)
          && this_node->GlobalWorkStreamId() == next_node->GlobalWorkStreamId()) {
        if (next_node->chain_id() == -1) {
          queued_nodes.push(next_node);
          visited_nodes.insert(next_node);
        } else {
          CHECK_EQ(next_node->chain_id(), this_chain_id);
        }
      }
    });
  }
}

std::function<TaskNode*(const std::string&)> MakeGetterTaskNode4SoleOpName(
    const HashSet<TaskNode*>& task_nodes) {
  auto op_name2task_nodes = std::make_shared<HashMap<std::string, HashSet<TaskNode*>>>();
  for (TaskNode* task_node : task_nodes) {
    if (task_node->exec_gph().node_num() == 1) {
      ExecNode* exec_node = task_node->exec_gph().SoleNode();
      CHECK((*op_name2task_nodes)[exec_node->op()->op_name()].emplace(task_node).second);
    }
  }
  return [op_name2task_nodes](const std::string& op_name) -> TaskNode* {
    const auto& iter = op_name2task_nodes->find(op_name);
    if (iter == op_name2task_nodes->end()) { return nullptr; }
    if (iter->second.size() > 1) { return nullptr; }
    return *iter->second.begin();
  };
}

bool IsLbiOnTaskEdge(const TaskEdge* edge, const LogicalBlobId& lbi) {
  for (const auto& regst_desc : edge->GetRegsts()) {
    if (regst_desc->HasLbi(lbi)) { return true; }
  }
  return false;
}

std::function<bool(const LogicalBlobId&, const std::string&)>
MakePredicatorIsLbiAllConsumersReachable(
    const std::function<const TaskNode*(const std::string&)>& TaskNode4SoleOpName,
    const std::function<bool(const std::string&, const std::string&)>&
        IsOpNameDataOrCtrlReachable) {
  auto IsDataOrCtrlReachable = [IsOpNameDataOrCtrlReachable](const TaskNode* src_node,
                                                             const TaskNode* dst_node) -> bool {
    if (src_node->chain_id() == dst_node->chain_id()
        && src_node->order_in_graph() <= dst_node->order_in_graph()) {
      return true;
    }
    const CompTaskNode* comp_src_node = dynamic_cast<const CompTaskNode*>(src_node);
    if (comp_src_node == nullptr) { return false; }
    if (comp_src_node->logical_node()->op_vec().size() != 1) { return false; }
    const CompTaskNode* comp_dst_node = dynamic_cast<const CompTaskNode*>(dst_node);
    if (comp_dst_node == nullptr) { return false; }
    if (comp_dst_node->logical_node()->op_vec().size() != 1) { return false; }
    return IsOpNameDataOrCtrlReachable(comp_src_node->logical_node()->SoleOp()->op_name(),
                                       comp_dst_node->logical_node()->SoleOp()->op_name());
  };
  return [TaskNode4SoleOpName, IsDataOrCtrlReachable](const LogicalBlobId& lbi,
                                                      const std::string& op_name) -> bool {
    const TaskNode* src_task_node = TaskNode4SoleOpName(lbi.op_name());
    const TaskNode* dst_task_node = TaskNode4SoleOpName(op_name);
    size_t out_edges_size = 0;
    size_t reachable_out_edges_size = 0;
    for (TaskEdge* out_edge : src_task_node->out_edges()) {
      if (IsLbiOnTaskEdge(out_edge, lbi)) {
        out_edges_size += 1;
        reachable_out_edges_size += IsDataOrCtrlReachable(out_edge->dst_node(), dst_task_node);
      }
    }
    return out_edges_size > 0 && out_edges_size == reachable_out_edges_size;
  };
}

bool IsInplaceAllowed(
    TaskNode* task_node, const std::vector<std::string>& bns,
    const std::function<const TaskNode*(const std::string&)>& TaskNode4SoleOpName) {
  if (task_node->exec_gph().node_num() != 1) { return false; }
  const auto& exec_node = *task_node->exec_gph().SoleNode();
  for (const auto& bn : bns) {
    // TaskNode for bn is not nullptr if it's on the same device with `task_node`
    if (TaskNode4SoleOpName(exec_node.op()->BnInOp2Lbi(bn).op_name()) == nullptr) { return false; }
    const RegstDesc& regst_desc = *exec_node.RegstDesc4BnInOp(bn);
    if (regst_desc.NumOfLbi() != 1) { return false; }
  }
  const BlobDesc* first_blob = nullptr;
  for (const auto& bn : bns) {
    const BlobDesc* blob_desc = exec_node.RegstDesc4BnInOp(bn)->SoleBlobDesc();
    if (first_blob == nullptr) {
      first_blob = blob_desc;
    } else {
      if (!(first_blob->shape().elem_cnt() == blob_desc->shape().elem_cnt()
            && first_blob->data_type() == blob_desc->data_type())) {
        return false;
      }
    }
  }
  return true;
}

std::unique_ptr<BoxingLogger> CreateBoxingLogger() {
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    return std::unique_ptr<BoxingLogger>(
        new CsvBoxingLogger(StrCat("boxing/log/", GlobalJobDesc().job_id()) + ".csv"));
  } else {
    return std::unique_ptr<BoxingLogger>(new NullBoxingLogger());
  }
}

Maybe<void> MakeGetterTaskNode4MachineId7ThrdId(
    const std::vector<CompTaskNode*>& task_nodes,
    std::function<Maybe<CompTaskNode*>(int64_t mchn_id, int64_t thrd_id)>* Getter) {
  // ticks are shared within a machine/process
  auto machine_id2task_node = std::make_shared<HashMap<int64_t, CompTaskNode*>>();
  for (auto* task_node : task_nodes) {
    machine_id2task_node->emplace(task_node->machine_id(), task_node);
  }
  *Getter = [machine_id2task_node](int64_t mchn_id, int64_t thrd_id) -> Maybe<CompTaskNode*> {
    const auto& iter = machine_id2task_node->find(mchn_id);
    CHECK_OR_RETURN(iter != machine_id2task_node->end());
    return iter->second;
  };
  return Maybe<void>::Ok();
}

}  // namespace

TaskGraph::TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph) {
  logical_gph_ = std::move(logical_gph);
  sub_tsk_gph_builder_ctx_.reset(new SubTskGphBuilderCtx(this));
  boxing_logger_ = CreateBoxingLogger();
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  if (!Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  }
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  sub_tsk_gph_builder_.reset(new ChainSubTskGphBuilder(builders));
  HashMap<const LogicalNode*, std::vector<CompTaskNode*>> logical2sorted_comp_tasks;
  HashMap<CompTaskNode*, HashMap<std::pair<int64_t, MemZoneId>, TaskNode*>> buf_task;
  MutBufTaskFn MutBufTask = [&](CompTaskNode* task_node, int64_t machine_id,
                                MemZoneId mem_zone_id) -> TaskNode** {
    auto key = std::make_pair(machine_id, mem_zone_id);
    auto& task_map = buf_task[task_node];
    if (task_map.find(key) == task_map.end()) { task_map.emplace(key, nullptr); }
    return &task_map[key];
  };

  logical_gph_->ForEachNode([&](const LogicalNode* logical_node) {
    logical_node->GenSortedCompTaskNodes([&](CompTaskNode* comp_task_node) {
      AddAllocatedNode(comp_task_node);
      logical2sorted_comp_tasks[logical_node].push_back(comp_task_node);
    });
  });

  logical_gph_->ForEachEdge([&](const LogicalEdge* logical_edge) {
    BldSubTskGphMthd method =
        GetMthdForBldSubTskGph(logical_edge->src_node(), logical_edge->dst_node());
    (this->*method)(logical_edge->src_node(), logical_edge->dst_node(),
                    logical2sorted_comp_tasks.at(logical_edge->src_node()),
                    logical2sorted_comp_tasks.at(logical_edge->dst_node()), MutBufTask);
  });
  logical_gph_->ForEachNecessaryCtrlEdge(
      [&](const LogicalNode* src, const LogicalNode* dst, int64_t ctrl_regst_num) {
        const auto& src_task_nodes = logical2sorted_comp_tasks.at(src);
        const auto& dst_task_nodes = logical2sorted_comp_tasks.at(dst);
        if (src->SoleOp()->op_conf().has_src_subset_tick_conf()) {
          UNIMPLEMENTED();
        } else if (dst->SoleOp()->op_conf().has_dst_subset_tick_conf()) {
          UNIMPLEMENTED();
        } else {
          ConnectCtrlEdges(src_task_nodes, dst_task_nodes, ctrl_regst_num);
        }
      });

  SetOrderInGraphForEachNode();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { ToDotWithAutoFilePath(); }
}

Maybe<void> TaskGraph::ConnectDstSubsetTickEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                                 const std::vector<CompTaskNode*>& dst_task_nodes) {
  std::function<Maybe<CompTaskNode*>(int64_t mchn_id, int64_t thrd_id)> TaskNode4MachineId7ThrdId;
  JUST(MakeGetterTaskNode4MachineId7ThrdId(dst_task_nodes, &TaskNode4MachineId7ThrdId));
  for (CompTaskNode* src_task_node : src_task_nodes) {
    CompTaskNode* dst_task_node =
        JUST(TaskNode4MachineId7ThrdId(src_task_node->machine_id(), src_task_node->thrd_id()));
    TaskEdge* edge = NewEdge();
    Connect<TaskNode>(src_task_node, edge, dst_task_node);
  }
  return Maybe<void>::Ok();
}

void TaskGraph::ConnectCtrlEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                 const std::vector<CompTaskNode*>& dst_task_nodes,
                                 int64_t ctrl_regst_num) {
  CHECK_EQ(src_task_nodes.size(), dst_task_nodes.size());
  FOR_RANGE(int32_t, i, 0, src_task_nodes.size()) {
    std::string regst_desc_name;
    RegstDesc* ctrl_regst_desc =
        src_task_nodes.at(i)->BuildCtrlRegstDesc(dst_task_nodes.at(i), &regst_desc_name);
    ctrl_regst_desc->UpdtMinRegstNumIfNeed(ctrl_regst_num);
    ctrl_regst_desc->UpdtMaxRegstNumIfNeed(ctrl_regst_num);
    ctrl_regst_desc->mut_regst_desc_type()->mutable_ctrl_regst_desc()->set_returned_regst_num(
        ctrl_regst_num);

    TaskEdge* edge = NewEdge();
    Connect<TaskNode>(src_task_nodes.at(i), edge, dst_task_nodes.at(i));
    src_task_nodes.at(i)->BindEdgeWithProducedRegst(edge, regst_desc_name);
  }
}

void TaskGraph::RemoveEmptyRegsts() {
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedBlob(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeConsumedRegst(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedRegst(); });
  ForEachNode([&](TaskNode* node) { node->UnbindBnWithEmptyRegst(); });
}

void TaskGraph::MergeChainAndAddOrderingCtrlEdgeInSameChain() {
  MergeChain();
  BuildCtrlRegstDescInSameChain();
}

void TaskGraph::SetOrderInGraphForEachNode() {
  int64_t order_in_graph = 0;
  auto SetOrderInGraph = [&](TaskNode* task_node) {
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes_.emplace_back(task_node);
    ++order_in_graph;
  };
  TopoForEachNode(SetOrderInGraph);
}

void TaskGraph::MergeChain() {
  int64_t chain_id = 0;
  for (auto* this_node : ordered_task_nodes_) {
    // skip if this node has been set in a chain.
    if (this_node->chain_id() != -1) { continue; }

    CHECK_EQ(this_node->chain_id(), -1);
    if (CanBeMergedInChain(this_node)) {
      TraverseConnectedSubGraphMergeInThisChain(this_node, chain_id);
    } else {
      this_node->set_chain_id(chain_id);
    }

    ++chain_id;
  }
  for (auto* node : ordered_task_nodes_) { CHECK_NE(node->chain_id(), -1); }
}

void TaskGraph::BuildCtrlRegstDescInSameChain() {
  HashMap<int64_t, TaskNode*> chain_id2node;
  for (auto* node : ordered_task_nodes_) {
    if (IsConnectToTickOp(node)) { continue; }
    int64_t chain_id = node->chain_id();
    auto iter = chain_id2node.find(chain_id);
    if (iter == chain_id2node.end()) {
      CHECK(chain_id2node.emplace(chain_id, node).second);
    } else {
      TaskNode* src_node = iter->second;
      TaskNode* dst_node = node;
      std::string ctrl_regst_name;
      bool build_ctrl_edge = src_node->BuildCtrlRegstDescIfNeed(dst_node, &ctrl_regst_name);
      if (build_ctrl_edge) {
        CHECK(!ctrl_regst_name.empty());
        TaskEdge* edge = NewEdge();
        Connect<TaskNode>(src_node, edge, dst_node);
        src_node->BindEdgeWithProducedRegst(edge, ctrl_regst_name);
      }
      iter->second = dst_node;
    }
  }
}

void TaskGraph::GetInplaceOpBlobArgList(
    InplaceObasInfo* obas_info, const HashSet<TaskNode*>& dev_nodes,
    const std::function<const TaskNode*(const std::string&)>& TaskNode4OpName) const {
  auto AddMutableInplaceArgPair = [&](TaskNode* node, const std::string& ibn,
                                      const std::string& obn, const std::string& op_name) {
    if (IsInplaceAllowed(node, {ibn, obn}, TaskNode4OpName)) {
      auto* pair = obas_info->mut_inplace_oba_pairs.mutable_pair()->Add();
      *pair->mutable_first() = GenOpBlobArg(op_name, ibn);
      *pair->mutable_second() = GenOpBlobArg(op_name, obn);
    }
  };
  auto AddConstInplaceArgPair = [&](TaskNode* node, const std::string& ibn, const std::string& obn,
                                    const std::string& op_name) {
    if (IsInplaceAllowed(node, {ibn, obn}, TaskNode4OpName)) {
      auto* pair = obas_info->con_inplace_oba_pairs.mutable_pair()->Add();
      *pair->mutable_first() = GenOpBlobArg(op_name, ibn);
      *pair->mutable_second() = GenOpBlobArg(op_name, obn);
    }
  };

  for (TaskNode* task_node : dev_nodes) {
    if (task_node->exec_gph().node_num() != 1) { continue; }
    const auto& op = *task_node->exec_gph().SoleNode()->op();
    for (const std::string& ibn : op.input_bns()) {
      if (op.InputBlobModifier4Ibn(ibn).is_mutable()) {
        CHECK(IsInplaceAllowed(task_node, {ibn}, TaskNode4OpName));
        *obas_info->mut_in_obas.mutable_oba()->Add() = GenOpBlobArg(op.op_name(), ibn);
      }
    }
    for (const auto& pair : task_node->exec_gph().SoleNode()->mut_inplace_obn2ibn()) {
      AddMutableInplaceArgPair(task_node, pair.second, pair.first, op.op_name());
    }
    for (const auto& pair : task_node->exec_gph().SoleNode()->con_inplace_obn2ibn()) {
      AddConstInplaceArgPair(task_node, pair.second, pair.first, op.op_name());
    }
  }
}

void TaskGraph::GetSafeInplaceOpBlobArgList(
    InplaceObasInfo* safe_obas_info, const HashSet<TaskNode*>& dev_nodes,
    const std::function<bool(const std::string&, const std::string&)>& IsOpNameDataOrCtrlReachable)
    const {
  auto TaskNode4SoleOpName = MakeGetterTaskNode4SoleOpName(dev_nodes);
  InplaceObasInfo obas_info;
  GetInplaceOpBlobArgList(&obas_info, dev_nodes, TaskNode4SoleOpName);
  auto Op4OpName = [&](const std::string& op_name) -> const Operator* {
    return TaskNode4SoleOpName(op_name)->exec_gph().SoleNode()->op().get();
  };
  auto IsLbiAllConsumersReachable =
      MakePredicatorIsLbiAllConsumersReachable(TaskNode4SoleOpName, IsOpNameDataOrCtrlReachable);
  InplaceLbiGraph origin_graph(obas_info, Op4OpName);
  InplaceLbiGraph safe_graph(*safe_obas_info, Op4OpName);
  origin_graph.ComputeSafeInplaceObns(safe_obas_info, IsLbiAllConsumersReachable);
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    origin_graph.ToDotWithFilePath(
        JoinPath("dot", "InplaceLbiGraph", GlobalJobDesc().job_name() + "_origin.dot"));
    safe_graph.ToDotWithFilePath(
        JoinPath("dot", "InplaceLbiGraph", GlobalJobDesc().job_name() + "_safe.dot"));
  }
}

void TaskGraph::SetTaskRegstInplaceInfo(const InplaceObasInfo& obas_info,
                                        const HashSet<TaskNode*>& dev_nodes) const {
  auto TaskNode4SoleOpName = MakeGetterTaskNode4SoleOpName(dev_nodes);
  auto Op4OpName = [&](const std::string& op_name) -> const Operator* {
    return TaskNode4SoleOpName(op_name)->exec_gph().SoleNode()->op().get();
  };
  InplaceLbiGraph inplace_gph(obas_info, Op4OpName);
  inplace_gph.ForEachConnectedComponent([&](const HashSet<const InplaceLbiNode*>& inplace_nodes) {
    for (const auto* inplace_node : inplace_nodes) {
      if (inplace_node->in_edges().empty()) { continue; }
      const auto* inplace_edge = inplace_node->SoleInEdge();
      auto* exec_node = TaskNode4SoleOpName(inplace_edge->op().op_name())->exec_gph().SoleNode();
      RegstDesc* in_regst = exec_node->RegstDesc4BnInOp(inplace_edge->ibn());
      RegstDesc* out_regst = exec_node->RegstDesc4BnInOp(inplace_edge->obn());
      out_regst->set_hint_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
    }
  });
}

void TaskGraph::ForEachGpuDeviceNodes(
    const std::function<void(const HashSet<TaskNode*>& dev_nodes)>& Handler) const {
  HashMap<std::pair<int64_t, int64_t>, HashSet<TaskNode*>> global_dev_phy_id2nodes;
  ForEachNode([&](TaskNode* task_node) {
    if (task_node->device_type() != DeviceType::kGPU) { return; }
    int64_t dev_phy_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task_node->thrd_id());
    global_dev_phy_id2nodes[{task_node->machine_id(), dev_phy_id}].emplace(task_node);
  });
  for (const auto& pair : global_dev_phy_id2nodes) { Handler(pair.second); }
}

void TaskGraph::EnableInplaceMemSharing(
    const std::function<bool(const std::string&, const std::string&)>&
        IsOpNameDataOrCtrlReachable) {
  ForEachGpuDeviceNodes([&](const HashSet<TaskNode*>& dev_nodes) {
    InplaceObasInfo safe_inplace_obas_info;
    GetSafeInplaceOpBlobArgList(&safe_inplace_obas_info, dev_nodes, IsOpNameDataOrCtrlReachable);
    SetTaskRegstInplaceInfo(safe_inplace_obas_info, dev_nodes);
  });
}

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name) \
  void TaskGraph::method_name BLD_SUB_TSK_GPH_MTHD_ARGS()

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing) {
  const std::vector<LogicalBlobId> lbis = src_logical->GetLbisTo(dst_logical);
  for (const LogicalBlobId& lbi : lbis) {
    std::vector<TaskNode*> in_nodes;
    if (lbis.size() == 1) {
      in_nodes.assign(sorted_src_comp_tasks.begin(), sorted_src_comp_tasks.end());
    } else {
      for (CompTaskNode* src_node : sorted_src_comp_tasks) {
        auto* identity_node = NewNode<BoxingIdentityTaskNode>();
        identity_node->Init(src_node->machine_id(), src_node->thrd_id(), lbi);
        Connect<TaskNode>(src_node, NewEdge(), identity_node);
        in_nodes.push_back(identity_node);
      }
    }
    std::vector<TaskNode*> out_nodes;
    out_nodes.reserve(sorted_dst_comp_tasks.size());
    std::vector<std::vector<TaskNode*>> sorted_ctrl_tasks;
    const SbpParallel& src_sbp_parallel =
        Global<OpGraph>::Get()->GetSbpParallel(src_logical->SoleOp()->op_name(), lbi);
    const SbpParallel& dst_sbp_parallel =
        Global<OpGraph>::Get()->GetSbpParallel(dst_logical->SoleOp()->op_name(), lbi);
    const std::shared_ptr<const ParallelDesc>& src_parallel_desc = src_logical->parallel_desc();
    const std::shared_ptr<const ParallelDesc>& dst_parallel_desc = dst_logical->parallel_desc();
    const BlobDesc& blob_desc = Global<OpGraph>::Get()->GetLogicalBlobDesc(lbi);
    auto status = CHECK_JUST(sub_tsk_gph_builder_->Build(
        sub_tsk_gph_builder_ctx_.get(), in_nodes, &out_nodes, &sorted_ctrl_tasks,
        *src_parallel_desc, *dst_parallel_desc, lbi, blob_desc, src_sbp_parallel, dst_sbp_parallel,
        *src_logical->out_blob_time_shape()));
    boxing_logger_->Log(*status, src_logical->SoleOp()->op_name(), dst_logical->SoleOp()->op_name(),
                        *src_parallel_desc, *dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel,
                        lbi, blob_desc);
    sub_tsk_gph_builder_ctx_->ConnectAll121(out_nodes, sorted_dst_comp_tasks);
    if (!sorted_ctrl_tasks.empty()) {
      CHECK_EQ(sorted_ctrl_tasks.size(), sorted_dst_comp_tasks.size());
      FOR_RANGE(size_t, i, 0, sorted_dst_comp_tasks.size()) {
        for (TaskNode* ctrl_node : sorted_ctrl_tasks.at(i)) {
          Connect<TaskNode>(ctrl_node, NewEdge(), sorted_dst_comp_tasks.at(i));
          ctrl_node->BuildCtrlRegstDesc(sorted_dst_comp_tasks.at(i));
        }
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks.at(i);
    CompTaskNode* dst = sorted_dst_comp_tasks.at(i);
    BuildTaskPath(src, dst, MutBufTask, true);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast) {
  for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
    CompTaskNode* nearest_src_node =
        SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
    CHECK_NOTNULL(nearest_src_node);
    BuildTaskPath(nearest_src_node, dst_node, MutBufTask, true);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialInLbiConnect) {
  HashSet<LogicalBlobId> lbis;
  for (const auto& obn : src_logical->SoleOp()->output_bns()) {
    lbis.insert(src_logical->SoleOp()->BnInOp2Lbi(obn));
  }
  CHECK_EQ(sorted_src_comp_tasks.size(), 1);
  CHECK_EQ(dst_logical->SoleOp()->input_bns().size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_dst_comp_tasks.size()) {
    const auto& lbi = dst_logical->SoleOp()->BnInOp2Lbi(dst_logical->SoleOp()->input_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      BuildTaskPath(sorted_src_comp_tasks.at(0), sorted_dst_comp_tasks.at(i), MutBufTask, true);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialOutLbiConnect) {
  HashSet<LogicalBlobId> lbis;
  for (const auto& ibn : dst_logical->SoleOp()->input_bns()) {
    lbis.insert(dst_logical->SoleOp()->BnInOp2Lbi(ibn));
  }
  CHECK_EQ(sorted_dst_comp_tasks.size(), 1);
  CHECK_EQ(src_logical->SoleOp()->output_bns().size(), sorted_src_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_src_comp_tasks.size()) {
    const auto& lbi = src_logical->SoleOp()->BnInOp2Lbi(src_logical->SoleOp()->output_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(0), MutBufTask, true);
    }
  }
}

Maybe<void> TaskGraph::ConnectSrcSubsetTickEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                                 const std::vector<CompTaskNode*>& dst_task_nodes) {
  std::function<Maybe<CompTaskNode*>(int64_t mchn_id, int64_t thrd_id)> TaskNode4MachineId7ThrdId;
  JUST(MakeGetterTaskNode4MachineId7ThrdId(src_task_nodes, &TaskNode4MachineId7ThrdId));
  for (CompTaskNode* dst_task_node : dst_task_nodes) {
    CompTaskNode* src_task_node =
        JUST(TaskNode4MachineId7ThrdId(dst_task_node->machine_id(), dst_task_node->thrd_id()));
    TaskEdge* edge = NewEdge();
    Connect<TaskNode>(src_task_node, edge, dst_task_node);
  }
  return Maybe<void>::Ok();
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySrcSubsetConnect) {
  CHECK_JUST(ConnectSrcSubsetTickEdges(sorted_src_comp_tasks, sorted_dst_comp_tasks));
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByDstSubsetConnect) {
  CHECK_JUST(ConnectDstSubsetTickEdges(sorted_src_comp_tasks, sorted_dst_comp_tasks));
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphNormalForwardToDecodeH2D) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks.at(i);
    CompTaskNode* dst = sorted_dst_comp_tasks.at(i);
    Connect<TaskNode>(src, NewEdge(), dst);
  }
}

void TaskGraph::BuildTaskPath(CompTaskNode* src, CompTaskNode* dst, MutBufTaskFn MutBufTask,
                              bool use_buf_task_node) {
  CHECK_NE(src, dst);
  auto GetBufTask = [&](int64_t machine_id, MemZoneId mem_zone_id) -> TaskNode* {
    return *MutBufTask(src, machine_id, mem_zone_id);
  };
  auto SetBufTask = [&](int64_t machine_id, MemZoneId mem_zone_id, TaskNode* new_val) -> TaskNode* {
    TaskNode** cur_val = MutBufTask(src, machine_id, mem_zone_id);
    if (*cur_val == nullptr) {
      *cur_val = new_val;
    } else {
      CHECK_EQ(*cur_val, new_val);
    }
    return new_val;
  };
  TaskNode* cur_node = src;
  while (cur_node->machine_id() != dst->machine_id()
         || cur_node->MemZoneId121() != dst->MemZoneId121()) {
    cur_node = BuildTaskStep(cur_node, dst, GetBufTask, SetBufTask, use_buf_task_node);
  }
  if (cur_node != dst) { Connect<TaskNode>(cur_node, NewEdge(), dst); }
}

TaskNode* TaskGraph::BuildTaskStep(TaskNode* src, TaskNode* dst, const GetBufTaskFn& GetBufTask,
                                   const SetBufTaskFn& SetBufTask, bool use_buf_task_node) {
  MemZoneId next_mem_zone_id;
  TaskNode* next = nullptr;
  uint32_t src_process_id = static_cast<uint32_t>(src->machine_id());
  uint32_t dst_process_id = static_cast<uint32_t>(dst->machine_id());
  if (src->MemZoneId121().device_type() != DeviceType::kCPU) {
    next_mem_zone_id = MemZoneId(DeviceType::kCPU, 0);
    if (!use_buf_task_node || !(next = GetBufTask(src_process_id, next_mem_zone_id))) {
      next = AddCopyD2HTaskFrom(src);
      Connect<TaskNode>(src, NewEdge(), next);
    }
  } else if (src->machine_id() == dst->machine_id()) {
    next_mem_zone_id = dst->MemZoneId121();
    if (!use_buf_task_node || !(next = GetBufTask(src_process_id, next_mem_zone_id))) {
      next = TryAddCopyH2DTaskTo(dst);
      if (next == nullptr) { next = dst; }
      Connect<TaskNode>(src, NewEdge(), next);
    }
  } else if (src->machine_id() != dst->machine_id()) {
    next_mem_zone_id = MemZoneId(DeviceType::kCPU, 0);
    if (!use_buf_task_node || !(next = GetBufTask(dst_process_id, next_mem_zone_id))) {
      next = AddCopyCommNetTaskBetween(src, dst);
      Connect<TaskNode>(src, NewEdge(), next);
    }
  } else {
    UNIMPLEMENTED();
  }
  if (use_buf_task_node && (next != dst)) {
    uint32_t next_process_id = static_cast<uint32_t>(next->machine_id());
    SetBufTask(next_process_id, next_mem_zone_id, next);
  }
  return next;
}

TaskNode* TaskGraph::TryAddCopyH2DTaskTo(TaskNode* task) {
  if (IsInterfaceTask(task)) { return nullptr; }
  if (IsClassRegistered<int32_t, TickTockTaskType>(task->GetTaskType())) { return nullptr; }
  CHECK_NE(task->device_type(), DeviceType::kCPU);
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  StreamId stream_id = DeserializeStreamIdFromInt64(task->thrd_id());
  copy_task->Init(CopyHdOpConf::H2D, task->machine_id(), stream_id.device_id().device_type(),
                  stream_id.device_id().device_index());
  return copy_task;
}

TaskNode* TaskGraph::AddCopyD2HTaskFrom(TaskNode* task) {
  CHECK_NE(task->device_type(), DeviceType::kCPU);
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  StreamId stream_id = DeserializeStreamIdFromInt64(task->thrd_id());
  copy_task->Init(CopyHdOpConf::D2H, task->machine_id(), stream_id.device_id().device_type(),
                  stream_id.device_id().device_index());
  return copy_task;
}

TaskNode* TaskGraph::AddCopyCommNetTaskBetween(TaskNode* src, TaskNode* dst) {
  CHECK_NE(src->machine_id(), dst->machine_id());
  CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
  copy_comm_net_task->Init(dst->machine_id());
  return copy_comm_net_task;
}

void TaskGraph::ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst) {
  if (src->machine_id() == dst->machine_id()) {
    Connect(src, NewEdge(), dst);
  } else {
    TaskNode* copy_comm_net_task = AddCopyCommNetTaskBetween(src, dst);
    Connect<TaskNode>(src, NewEdge(), copy_comm_net_task);
    Connect<TaskNode>(copy_comm_net_task, NewEdge(), dst);
  }
}

}  // namespace oneflow

```

38. oneflow/core/graph/task_node.h
39. oneflow/core/graph/task_node.cpp
```.cpp
MemZoneId TaskNode::MemZoneId121() const {
  const auto task_id = DeserializeTaskIdFromInt64(task_id_);
  const DeviceId& device_id = task_id.stream_id().device_id();
  return MemZoneId(device_id.device_type(), device_id.device_index());
}

void TaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (device_type() == DeviceType::kCPU) {
    mem_case->mutable_host_mem();
  } else if (device_type() == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(
        Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id_));
  } else if (device_type() == DeviceType::kFAKEDEVICE) {
    mem_case->mutable_fake_dev_mem();
  } else if (device_type() == DeviceType::kCambricon) {
    mem_case->mutable_device_cambricon_mem()->set_device_id(0);
  } else {
    UNIMPLEMENTED();
  }
}
```

40. oneflow/core/graph_impl/normal_forward_compute_task_node.cpp 
```.cpp
#include "oneflow/core/device/fake_device_stream_index.h"
#include "oneflow/core/device/cambricon_device_stream_index.h"
```

41. oneflow/core/job/env_global_objects_scope.cpp
```.cpp
#ifdef WITH_CAMBRICON
#include "cnrt.h"
#endif  // WITH_CAMBRICON

#ifdef WITH_CAMBRICON
  CNRT_CHECK(cnrtInit(0));
#endif

#ifdef WITH_CAMBRICON
  cnrtDestroy();
#endif
```

42. oneflow/core/job/id_manager.h 
```.h
// MemZoneId
  int64_t CpuMemZoneId() const;
  bool IsCpuMemZone(int64_t mem_zone_id) const;
  bool IsGpuMemZone(int64_t mem_zone_id) const;
  int64_t GpuMemZoneId(int64_t dev_phy_id) const;
  int64_t GetGpuPhyIdFromMemZoneId(int64_t mem_zone_id) const;
```

43. oneflow/core/job/id_manager.cpp
```.cpp
int64_t IDMgr::CpuMemZoneId() const {
  MemZoneId cpu_mem_zone_id(DeviceType::kCPU, 0);
  return SerializeMemZoneIdToInt64(cpu_mem_zone_id);
}
bool IDMgr::IsCpuMemZone(int64_t mem_zone_id) const {
  MemZoneId cpu_mem_zone_id = DeserializeMemZoneIdFromInt64(mem_zone_id);
  return cpu_mem_zone_id.device_type() == DeviceType::kCPU;
}
bool IDMgr::IsGpuMemZone(int64_t mem_zone_id) const {
  MemZoneId gpu_mem_zone_id = DeserializeMemZoneIdFromInt64(mem_zone_id);
  return gpu_mem_zone_id.device_type() == DeviceType::kGPU;
}
int64_t IDMgr::GpuMemZoneId(int64_t dev_phy_id) const {
  MemZoneId gpu_mem_zone_id(DeviceType::kGPU, dev_phy_id);
  return SerializeMemZoneIdToInt64(gpu_mem_zone_id);
}
int64_t IDMgr::GetGpuPhyIdFromMemZoneId(int64_t mem_zone_id) const {
  CHECK_LT(mem_zone_id, gpu_device_num_);
  return mem_zone_id;
  MemZoneId gpu_mem_zone_id = DeserializeMemZoneIdFromInt64(mem_zone_id);
  return static_cast<int64_t>(gpu_mem_zone_id.device_index());
}
```

44. oneflow/core/job/improver.cpp
```.cpp
*(mem_block.mutable_mem_case()) =
          GenerateCorrespondingPageLockedHostMemoryCase(regst_desc->mem_case());
          
auto GenChunk4ReusedMemBlockIfNeed = [&](MemBlockProto* mem_block) {
    int64_t mzuid = SerializeGlobalMemCaseIdToInt64(GlobalMemCaseId{
        static_cast<GlobalMemCaseId::rank_t>(mem_block->machine_id()), mem_block->mem_case()});
```

45. oneflow/core/job/inter_job_mem_sharing_util.cpp
```.cpp
int64_t mzuid = SerializeGlobalMemCaseIdToInt64(
        GlobalMemCaseId{static_cast<GlobalMemCaseId::rank_t>(chunk.machine_id()), mem_case});
        
MemoryCase header_mem_case = GenerateCorrespondingPageLockedHostMemoryCase(lhs->mem_case());

CHECK(PatchMemCase(&common_mem_case_vec[i], regst_desc->mem_case()));
```

46. oneflow/core/job/plan_util.cpp
```.cpp
CHECK(header_mem_block.mem_case()
              == GenerateCorrespondingPageLockedHostMemoryCase(regst.mem_case()));
```

47. oneflow/core/job/resource.proto
```.proto
optional int32 mlu_device_num = 3 [default = 0];
```

48. oneflow/core/job_rewriter/add_input_output_ops_pass.cpp
```.cpp
// NOTE: return op is best to be run on cpu by this we could only implement cpu return kernel
  output_op_conf->set_device_tag("cpu");
  
HashMap<std::string, ParallelConf> io_op_name2parallel_conf;

CHECK_OR_RETURN(
        io_op_name2parallel_conf.emplace(input_name, op_node->parallel_desc().parallel_conf())
            .second);
            
// NOTE: Ouput op (return op) is preferably on host
    ParallelConf host_parallel_conf(op_node->parallel_desc().parallel_conf());
    host_parallel_conf.set_device_tag("cpu");
    CHECK_OR_RETURN(io_op_name2parallel_conf.emplace(output_name, host_parallel_conf).second);
  }
  
for (const auto& pair : io_op_name2op_conf) {
    job_builder->AddOps(io_op_name2parallel_conf.at(pair.first), {pair.second});
  }
```

49. 在 oneflow/core/kernel/copy_hd_kernel.cpp 增加 REGISTER_KERNEL_WITH_DEVICE 宏

oneflow/core/kernel/copy_hd_kernel.cpp
```.cpp
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class CopyHdKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
    Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
    out_blob->CopyValidDataContentFrom(ctx.device_ctx, in_blob);
  };
  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    BnInOp2Blob("out")->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob("in"));
  }
};

#ifdef WITH_CUDA

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCopyHdConf, DeviceType::kGPU,
                            CopyHdKernel<DeviceType::kGPU>);
#endif

#ifdef WITH_FAKE_DEVICE
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCopyHdConf, DeviceType::kFAKEDEVICE,
                            CopyHdKernel<DeviceType::kFAKEDEVICE>);
#endif

#ifdef WITH_CAMBRICON
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCopyHdConf, DeviceType::kCambricon,
                            CopyHdKernel<DeviceType::kCambricon>);
#endif
```

50. oneflow/core/kernel/device_tick_kernel.cpp 
```.cpp
ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE(OperatorConf::kDeviceTickConf, DeviceTickKernel);
```

51. oneflow/core/kernel/input_kernel.cpp
```.cpp
ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE(OperatorConf::kInputConf, InputKernel);
```

52. 只要修改 oneflow/core/kernel/kernel.h 里的两个宏就好了: `ADD_DEFAULT_KERNEL_CREATOR_FAKE` 和 `ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE`<br>
在以下四个文件中生效：
```
oneflow/core/kernel/device_tick_kernel.cpp 
oneflow/core/kernel/input_kernel.cpp
oneflow/core/kernel/output_kernel.cpp   增加 ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE
oneflow/core/kernel/variable_kernel.cpp 增加 ADD_DEFAULT_KERNEL_CREATOR_FAKE
```

oneflow/core/kernel/kernel.h
```.h
#ifdef WITH_CAMBRICON
#define CAMBRICON_TUPLE_SEQ OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCambricon)
#else
#define CAMBRICON_TUPLE_SEQ
#endif

#define FAKE_DEV_TUPLE_SEQ OF_PP_MAKE_TUPLE_SEQ(DeviceType::kFAKEDEVICE)

#define ADD_DEFAULT_KERNEL_CREATOR_FAKE(op_type_case, kernel_class, data_type_seq)               \
  namespace {                                                                                    \
                                                                                                 \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {                     \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {                     \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (kernel_class),              \
                                         DEVICE_TYPE_SEQ FAKE_DEV_TUPLE_SEQ CAMBRICON_TUPLE_SEQ, \
                                         data_type_seq)};                                        \
    DeviceType device_type =                                                                     \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag()));     \
    return creators.at(GetHashKey(device_type, kernel_conf.data_type()))();                      \
  }                                                                                              \
                                                                                                 \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));                      \
  }
  
  #define ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE(op_type_case, kernel_class)                \
  namespace {                                                                                    \
                                                                                                 \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {                     \
    static const HashMap<int, std::function<Kernel*()>> creators = {                             \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(                                                        \
            MAKE_DEVICE_TYPE_KERNEL_CREATOR_ENTRY, (kernel_class),                               \
            DEVICE_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(DeviceType::kFAKEDEVICE) CAMBRICON_TUPLE_SEQ)}; \
    DeviceType device_type =                                                                     \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag()));     \
    return creators.at(device_type)();                                                           \
  }                                                                                              \
                                                                                                 \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));                      \
  }
```

53. oneflow/core/kernel/kernel_util.cpp 
```.cpp
if (src_mem_case.has_host_mem() && dst_mem_case.has_host_mem()) {
    func = &Memcpy<DeviceType::kCPU>;
  } else if (src_mem_case.has_fake_dev_mem() || dst_mem_case.has_fake_dev_mem()) {
    func = &Memcpy<DeviceType::kFAKEDEVICE>;
  } else if (src_mem_case.has_device_cuda_mem() || dst_mem_case.has_device_cuda_mem()) {
#ifdef WITH_CUDA
    func = &Memcpy<DeviceType::kGPU>;
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  } else if (src_mem_case.has_device_cambricon_mem() || dst_mem_case.has_device_cambricon_mem()) {
#ifdef WITH_CAMBRICON
    if (src_mem_case.has_device_cambricon_mem() && !dst_mem_case.has_device_cambricon_mem()) {
      CNRT_CHECK(cnrtMemcpyAsync(dst, (void*)(src), sz, ctx->cambricon_queue(),
                                 CNRT_MEM_TRANS_DIR_DEV2HOST));
    } else if (!src_mem_case.has_device_cambricon_mem()
               && dst_mem_case.has_device_cambricon_mem()) {
      CNRT_CHECK(cnrtMemcpyAsync(dst, (void*)(src), sz, ctx->cambricon_queue(),
                                 CNRT_MEM_TRANS_DIR_HOST2DEV));
    } else if (src_mem_case.has_device_cambricon_mem() && dst_mem_case.has_device_cambricon_mem()) {
      CNRT_CHECK(cnrtMemcpyAsync(dst, (void*)(src), sz, ctx->cambricon_queue(),
                                 CNRT_MEM_TRANS_DIR_DEV2DEV));
    } else {
      UNIMPLEMENTED();
    }
    return;
#else
    UNIMPLEMENTED();
#endif  // WITH_CAMBRICON
  }
```

54. oneflow/core/kernel/matmul_kernel_mlu.cpp
```.cpp
#ifdef WITH_CAMBRICON

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/mlu_tools.h"
#include "cnrt.h"
#include "cnnl.h"
#include <stdio.h>

namespace oneflow {

typedef struct Matmul_ {
  cnnlTensorDescriptor_t a_desc = nullptr;
  cnnlTensorDescriptor_t b_desc = nullptr;
  cnnlTensorDescriptor_t c_desc = nullptr;
} Matmul;

template<DeviceType device_type, CamDataType T>
class MatmulKernelCambricon final : public user_op::OpKernel {
 public:
  MatmulKernelCambricon() = default;
  ~MatmulKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");
    user_op::Tensor* c = ctx->Tensor4ArgNameAndIndex("out", 0);

    void* a_ptr = (void*)a->dptr();
    void* b_ptr = (void*)b->dptr();
    void* c_ptr = (void*)c->dptr();

    Matmul matmul;
    MatMulType datainfo;
    datainfo.input_dtype = convert(CamDataType::kINT8);
    datainfo.output_dtype = convert(T);

    CHECK_EQ(a->shape().NumAxes(), 2);
    CHECK_EQ(b->shape().NumAxes(), 2);

    set_tensor_desc_matmul(matmul.a_desc, a->shape().At(0), a->shape().At(1), CNNL_DTYPE_INT8,
                           datainfo.layout);

    set_tensor_desc_matmul(matmul.b_desc, b->shape().At(0), b->shape().At(1), CNNL_DTYPE_INT8,
                           datainfo.layout);

    set_tensor_desc_matmul(matmul.c_desc, c->shape().At(0), c->shape().At(1), datainfo.output_dtype,
                           datainfo.layout);
    // cast a
    void* a_cast;
    int a_size = a->shape().elem_cnt();
    CNRT_CHECK(cnrtMalloc(&(a_cast), a_size));
    CNRT_CHECK(cnrtMemset(a_cast, 0, a_size));

    void* a_pos = nullptr;
    void* a_scale = nullptr;
    CNRT_CHECK(cnrtMalloc((void**)&a_pos, sizeof(int32_t)));
    CNRT_CHECK(cnrtMalloc((void**)&a_scale, sizeof(float)));
    size_t a_workspace_size = 0;
    void* a_workspace = nullptr;
    cnnlTensorDescriptor_t a_desc = nullptr;
    set_tensor_desc_matmul(a_desc, a->shape().At(0), a->shape().At(1), CNNL_DTYPE_FLOAT,
                           datainfo.layout);

    CNNL_CHECK(cnnlGetQuantifyOnlineWorkspaceSize(ctx->device_ctx()->cambricon_handle(), a_desc,
                                                  &a_workspace_size));
    if (a_workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&a_workspace, a_workspace_size));
      CNRT_CHECK(cnrtMemset(a_workspace, 0, a_workspace_size));
    }
    cnnlQuantifyOnline(ctx->device_ctx()->cambricon_handle(), false, a_desc, a_ptr, a_workspace,
                       a_workspace_size, a_pos, a_scale, matmul.a_desc, a_cast);
    ctx->device_ctx()->SyncDevice();
    int a_pos_ = 0;
    float a_scale_ = 0;
    CNRT_CHECK(cnrtMemcpy(&a_pos_, a_pos, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&a_scale_, a_scale, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnnlSetTensorDescriptorPositionAndScale(matmul.a_desc, a_pos_, a_scale_);

    // cast b
    void* b_cast;
    int b_size = b->shape().elem_cnt();
    CNRT_CHECK(cnrtMalloc(&(b_cast), b_size));

    void* b_pos = nullptr;
    void* b_scale = nullptr;
    CNRT_CHECK(cnrtMalloc((void**)&b_pos, sizeof(int32_t)));
    CNRT_CHECK(cnrtMalloc((void**)&b_scale, sizeof(float)));
    size_t b_workspace_size = 0;
    void* b_workspace = nullptr;
    cnnlTensorDescriptor_t b_desc = nullptr;
    set_tensor_desc_matmul(b_desc, b->shape().At(0), b->shape().At(1), CNNL_DTYPE_FLOAT,
                           datainfo.layout);

    CNNL_CHECK(cnnlGetQuantifyOnlineWorkspaceSize(ctx->device_ctx()->cambricon_handle(), b_desc,
                                                  &b_workspace_size));
    if (b_workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc((void**)&b_workspace, b_workspace_size));
      CNRT_CHECK(cnrtMemset(b_workspace, 0, b_workspace_size));
    }
    cnnlQuantifyOnline(ctx->device_ctx()->cambricon_handle(), false, b_desc, b_ptr, b_workspace,
                       b_workspace_size, b_pos, b_scale, matmul.b_desc, b_cast);
    ctx->device_ctx()->SyncDevice();
    int b_pos_ = 0;
    float b_scale_ = 0;
    CNRT_CHECK(cnrtMemcpy(&b_pos_, b_pos, sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&b_scale_, b_scale, sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    cnnlSetTensorDescriptorPositionAndScale(matmul.b_desc, b_pos_, b_scale_);

    bool is_trans_a = transpose_a;
    bool is_trans_b = transpose_b;
    float* alpha = (float*)malloc(1 * sizeof(float));
    alpha[0] = 1.0;
    float* beta = (float*)malloc(1 * sizeof(float));
    beta[0] = 0.0;

    CNNL_CHECK(cnnlMatMul(ctx->device_ctx()->cambricon_handle(), is_trans_a, is_trans_b,
                          (void*)alpha, matmul.a_desc, a_cast, matmul.b_desc, b_cast, (void*)beta,
                          matmul.c_desc, c_ptr));

    if (a_workspace != nullptr) { CNRT_CHECK(cnrtFree(a_workspace)); }
    if (b_workspace != nullptr) { CNRT_CHECK(cnrtFree(b_workspace)); }
    CNRT_CHECK(cnrtFree(a_pos));
    CNRT_CHECK(cnrtFree(a_scale));
    CNRT_CHECK(cnrtFree(b_pos));
    CNRT_CHECK(cnrtFree(b_scale));
    CNRT_CHECK(cnrtFree(a_cast));
    CNRT_CHECK(cnrtFree(b_cast));
    free(alpha);
    free(beta);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATMUL_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("matmul")                           \
      .SetCreateFn<MatmulKernelCambricon<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));

REGISTER_MATMUL_KERNEL(DeviceType::kCambricon, CamDataType::kFLOAT32)

}  // namespace oneflow

#endif

```

55. oneflow/core/kernel/mlu_tools.h
```.h
#ifndef ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_
#define ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_

#ifdef WITH_CAMBRICON

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

#include "cnrt.h"
#include "cnnl.h"

namespace oneflow {

enum CamDataType { kHALF, kFLOAT32, kINT32, kINT16, kINT8 };

struct convDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t weight_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct poolDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct reluDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct sigmoidDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct preluDataType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t alpha_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct BatchNormType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t weight_bias_mean_var_desc_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct InstanceNormType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlDataType_t scale_bias_desc_dtype;
  cnnlDataType_t mean_var_desc_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct SoftmaxType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct MatMulType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct TransposeType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct InterpType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct AddType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct ConcatType {
  cnnlDataType_t input_dtype;
  cnnlDataType_t output_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

struct BiasAddType {
  cnnlDataType_t a_dtype;
  cnnlDataType_t b_dtype;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
};

size_t dataSize(cnnlDataType_t dtype);

void set_tensor_desc(cnnlTensorDescriptor_t& desc, size_t ndim, const int* dim,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc(cnnlTensorDescriptor_t& desc, int dim_n, int dim_h, int dim_w, int dim_c,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_3d(cnnlTensorDescriptor_t& desc, int dim_0, int dim_1, int dim_2,
                        cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_batchnorm(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                               cnnlTensorLayout_t layout);

void set_tensor_desc_softmax(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                             cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_matmul(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                            cnnlDataType_t dtype, cnnlTensorLayout_t layout);

void set_tensor_desc_biasadd(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                             cnnlTensorLayout_t layout);

cnrtDataType_t convertCnnlDtypeToCnrt(cnnlDataType_t dtype);

void getPosition(float* input, size_t num, cnnlDataType_t datatype, int* position);

void getPositionAndScale(float* input, size_t size, cnnlDataType_t dtype, int* pos, float* scale);

void cast_data(float* src_data, cnnlDataType_t src_dtype, char* dst_data, cnnlDataType_t dst_dtype,
               size_t size, int* pos, float* scale, int* offset, int quant_mode);

cnnlDataType_t convert(CamDataType type);

}  // namespace oneflow

#endif

#endif  // ONEFLOW_CORE_KERNEL_MLU_TOOLS_H_

```

56. oneflow/core/kernel/mlu_tools.cpp
```.cpp
/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef WITH_CAMBRICON

#include <stdexcept>
#include "oneflow/core/kernel/mlu_tools.h"

namespace oneflow {

size_t dataSize(cnnlDataType_t dtype) {
  switch (dtype) {
    case CNNL_DTYPE_HALF: return 2;
    case CNNL_DTYPE_FLOAT: return 4;
    case CNNL_DTYPE_INT8: return 1;
    case CNNL_DTYPE_INT16: return 2;
    case CNNL_DTYPE_INT32: return 4;
    default: throw std::runtime_error("unsupport data  dtype\n");
  }
}

cnnlDataType_t convert(CamDataType type) {
  int v = 0;
  if (type == kHALF) {
    v = 1;
  } else if (type == kFLOAT32) {
    v = 2;
  } else if (type == kINT8) {
    v = 3;
  } else if (type == kINT16) {
    v = 4;
  }
  return (cnnlDataType_t)v;
}

void set_tensor_desc(cnnlTensorDescriptor_t& desc, size_t ndim, const int* dim,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, static_cast<int>(ndim), dim));
}

void set_tensor_desc(cnnlTensorDescriptor_t& desc, int dim_n, int dim_h, int dim_w, int dim_c,
                     cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[4];
  dim[0] = dim_n;
  dim[1] = dim_h;
  dim[2] = dim_w;
  dim[3] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 4, dim));
}

void set_tensor_desc_3d(cnnlTensorDescriptor_t& desc, int dim_0, int dim_1, int dim_2,
                        cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[3];
  dim[0] = dim_0;
  dim[1] = dim_1;
  dim[2] = dim_2;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 3, dim));
}

void set_tensor_desc_batchnorm(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                               cnnlTensorLayout_t layout) {
  int dim[1];
  dim[0] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 1, dim));
}

void set_tensor_desc_softmax(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                             cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[3];
  dim[0] = dim_n;
  dim[1] = 1;
  dim[2] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 3, dim));
}

void set_tensor_desc_matmul(cnnlTensorDescriptor_t& desc, int dim_n, int dim_c,
                            cnnlDataType_t dtype, cnnlTensorLayout_t layout) {
  int dim[2];
  dim[0] = dim_n;
  dim[1] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 2, dim));
}

void set_tensor_desc_biasadd(cnnlTensorDescriptor_t& desc, int dim_c, cnnlDataType_t dtype,
                             cnnlTensorLayout_t layout) {
  int dim[1];
  dim[0] = dim_c;
  CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  CNNL_CHECK(cnnlSetTensorDescriptor(desc, layout, dtype, 1, dim));
}

cnrtDataType_t convertCnnlDtypeToCnrt(cnnlDataType_t dtype) {
  switch (dtype) {
    case CNNL_DTYPE_HALF: return CNRT_FLOAT16;
    case CNNL_DTYPE_FLOAT: return CNRT_FLOAT32;
    case CNNL_DTYPE_INT8: return CNRT_INT8;
    case CNNL_DTYPE_INT16: return CNRT_INT16;
    case CNNL_DTYPE_INT32: return CNRT_INT32;
    case CNNL_DTYPE_BOOL: return CNRT_BOOL;
    case CNNL_DTYPE_UINT8: return CNRT_UINT8;
    default: throw std::runtime_error("unsupport data dtype\n");
  }
}

void getPosition(float* input, size_t num, cnnlDataType_t datatype, int* position) {
  if (input == nullptr || position == nullptr || num <= 0) { printf("invalid input paramter!\n"); }

  int bitwidth = 8;
  if (datatype == CNNL_DTYPE_INT8) {
    bitwidth = 8;
  } else if (datatype == CNNL_DTYPE_INT16) {
    bitwidth = 16;
  } else {
    printf("unsuport quant datatype!\n");
  }
  // Formula: int8 int16,  position = ceil(log2(absmax/(2^(bitwidth-1)-1)))
  float absmax = std::fabs(input[0]);
  for (size_t index = 0; index < num; ++index) {
    if (std::fabs(input[index]) > absmax) absmax = std::fabs(input[index]);
  }
  if (absmax == 0) {
    *position = 0;
  } else {
    *position = static_cast<int>(std::floor(std::log2(absmax)) - (bitwidth - 2));
  }
}

void getPositionAndScale(float* input, size_t size, cnnlDataType_t dtype, int* position,
                         float* scale, int mode) {
  if (input == NULL || size == 0 || position == NULL || scale == NULL) {
    printf("invalid input paramter!");
  }

  int bitwidth = 8;
  if (dtype == CNNL_DTYPE_INT8) {
    bitwidth = 8;
  } else if (dtype == CNNL_DTYPE_INT16) {
    bitwidth = 16;
  } else {
    printf("unsupport input data type!");
    return;
  }

  int scale_var = std::pow(2, bitwidth - 1) - 1;
  float max_data = std::fabs(input[0]);
  for (size_t index = 0; index < size; ++index) {
    if (std::fabs(input[index]) > max_data) max_data = std::fabs(input[index]);
  }
  if (max_data == 0) {
    *position = 0;
    *scale = 1.0;
  } else {
    if (mode == 0) {
      *position = static_cast<int>(std::floor(std::log2(max_data)) - (bitwidth - 2));
      *scale = static_cast<float>(std::pow(2, *position) * scale_var / max_data);
    } else {
      *position = static_cast<int>(std::floor(std::log2(max_data)) - (bitwidth - 2));
      *scale = 1.0;
    }
  }
}

void cast_data(float* src_data, cnnlDataType_t src_dtype, char* dst_data, cnnlDataType_t dst_dtype,
               size_t size, int* pos, float* scale, int* offset, int quant_mode) {
  if (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_FLOAT) {
    memcpy(dst_data, src_data, size * sizeof(float));
  } else if ((src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_INT8)
             || (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_INT16)) {
    auto in_dtype = convertCnnlDtypeToCnrt(src_dtype);
    auto out_dtype = convertCnnlDtypeToCnrt(dst_dtype);
    // need quant
    *pos = 0;
    *scale = 1.0;
    *offset = 0;

    if (0 == quant_mode) {
      getPosition(src_data, size, dst_dtype, pos);
    } else if (1 == quant_mode) {
      getPositionAndScale(src_data, size, dst_dtype, pos, scale, 1);
    } else {
      printf("This quant mode is not supported at present.");
    }
    cnrtQuantizedParam_t quant_param = nullptr;
    CNRT_CHECK(cnrtCreateQuantizedParam(&quant_param, *pos, *scale, *offset));
    CNRT_CHECK(cnrtCastDataType(src_data, in_dtype, dst_data, out_dtype, size, quant_param));
    CNRT_CHECK(cnrtDestroyQuantizedParam(quant_param));
  } else if ((src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_HALF)
             || (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_INT32)
             || (src_dtype == CNNL_DTYPE_FLOAT && dst_dtype == CNNL_DTYPE_BOOL)) {
    auto in_dtype = convertCnnlDtypeToCnrt(src_dtype);
    auto out_dtype = convertCnnlDtypeToCnrt(dst_dtype);
    CNRT_CHECK(cnrtCastDataType(src_data, in_dtype, dst_data, out_dtype, size, nullptr));
  } else {
    printf("This dtype is not supported.\n");
  }
}

}  // namespace oneflow

#endif

```

57. oneflow/core/kernel/new_kernel_util.cpp
```.cpp
#include "oneflow/core/framework/device_register_fakedev.h"

template<>
void Memcpy<DeviceType::kFAKEDEVICE>(DeviceCtx* ctx, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  memcpy(dst, src, sz);
}

template<>
void Memset<DeviceType::kFAKEDEVICE>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  memset(dst, value, sz);
}
```

58.  oneflow/core/kernel/new_kernel_util_mlu.cpp
```.cpp
#ifdef WITH_CAMBRICON

#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/blob.h"
#include "cnrt.h"

namespace oneflow {

template<>
void Memset<DeviceType::kCambricon>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  CNRT_CHECK(cnrtMemsetD8Async(dst, value, sz, ctx->cambricon_queue()));
}

template<>
void Memcpy<DeviceType::kCambricon>(DeviceCtx* ctx, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  ctx->SyncDevice();
  CNRT_CHECK(cnrtMemcpy(dst, (void*)(src), sz, CNRT_MEM_TRANS_DIR_NODIR));
}

}  // namespace oneflow

#endif
```

59. oneflow/core/kernel/output_kernel.cpp
```.cpp
ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE(OperatorConf::kOutputConf, OutputKernel);
```

60. oneflow/core/kernel/softmax_kernel_mlu.cpp
```.cpp
#ifdef WITH_CAMBRICON

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/mlu_tools.h"
#include "cnrt.h"
#include "cnnl.h"
#include <stdio.h>

namespace oneflow {

namespace {
typedef struct Softmax_ {
  cnnlTensorDescriptor_t input_desc = nullptr;
  cnnlTensorDescriptor_t output_desc = nullptr;
  cnnlSoftmaxAlgorithm_t algorithm = CNNL_SOFTMAX_ACCURATE;
  cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  float hw_time = 0;
} Softmax;

template<DeviceType device_type>
class SoftmaxKernelCambricon final : public user_op::OpKernel {
 public:
  SoftmaxKernelCambricon() = default;
  ~SoftmaxKernelCambricon() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);

    Softmax softmax;
    SoftmaxType datainfo;
    datainfo.input_dtype = convert(CamDataType::kFLOAT32);
    datainfo.output_dtype = convert(CamDataType::kFLOAT32);
    CHECK_EQ(x->shape().NumAxes(), 2)
        << "The number of axes of softmax op input shape should equal to 2!";
    set_tensor_desc_softmax(softmax.input_desc, x->shape().At(0), x->shape().At(1),
                            datainfo.input_dtype, datainfo.layout);
    set_tensor_desc_softmax(softmax.output_desc, y->shape().At(0), y->shape().At(1),
                            datainfo.output_dtype, datainfo.layout);

    const void* x_ptr = (const void*)x->dptr();
    void* y_ptr = (void*)y->dptr();

    CNNL_CHECK(cnnlSoftmaxForward(ctx->device_ctx()->cambricon_handle(), softmax.algorithm,
                                  softmax.mode, nullptr, softmax.input_desc, x_ptr, nullptr,
                                  softmax.output_desc, y_ptr));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_KERNEL(device, dtype)                                                  \
  REGISTER_USER_KERNEL("softmax")                                                               \
      .SetCreateFn<SoftmaxKernelCambricon<device>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("in", 0) == dtype))                              \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      })

REGISTER_SOFTMAX_KERNEL(DeviceType::kCambricon, DataType::kFloat);

}  // namespace

}  // namespace oneflow

#endif

```

61. oneflow/core/kernel/variable_kernel.cpp 
```.cpp
ADD_DEFAULT_KERNEL_CREATOR_FAKE(OperatorConf::kVariableConf, VariableKernel,
                                ARITHMETIC_DATA_TYPE_SEQ);
```

62. oneflow/core/memory/memory_allocator.cpp
```.cpp
#include "oneflow/core/memory/memory_fake_dev_allocator.h"
#ifdef WITH_CAMBRICON
#include "oneflow/core/device/mlu_util.h"
#endif


} else if (mem_case.has_fake_dev_mem()) {
    ptr = FakeDevMemoryAllocatorImpl::Allocate(mem_case, size);
  } else if (mem_case.has_device_cambricon_mem()) {
#ifdef WITH_CAMBRICON
    MLUCurrentDeviceGuard guard(mem_case.device_cambricon_mem().device_id());
    CNRT_CHECK(cnrtMalloc(&ptr, size));
#else
    UNIMPLEMENTED();
#endif  // WITH_CAMBRICON


} else if (mem_case.has_fake_dev_mem()) {
    FakeDevMemoryAllocatorImpl::Deallocate(ptr, mem_case);
  } else if (mem_case.has_device_cambricon_mem()) {
#ifdef WITH_CAMBRICON
    CNRT_CHECK(cnrtFree(ptr));
#else
    UNIMPLEMENTED();
#endif  // WITH_CAMBRICON


  } else if (mem_case.has_fake_dev_mem()) {
    memset(dptr, memset_val, size);
  } else if (mem_case.has_device_cambricon_mem()) {
#ifdef WITH_CAMBRICON
    CNRT_CHECK(cnrtMemset(dptr, memset_val, size));
#else
    UNIMPLEMENTED();
#endif  // WITH_CAMBRICON
```

63. oneflow/core/memory/memory_case.proto
```.proto
message FakeDevicePinnedMemory {
  optional int64 reserved = 1;
}


message HostMemory {
  oneof page_lock_case {
    CudaPinnedMemory cuda_pinned_mem = 1;
    FakeDevicePinnedMemory fake_dev_pinned_mem = 3;
  }
  
  
message DeviceCambriconMemory {
  required int64 device_id = 1;
}

message FakeDeviceMemory {
  optional int64 reserved = 1;
}


message MemoryCase {
  oneof case {
    HostMemory host_mem = 1;
    DeviceCudaMemory device_cuda_mem = 2;
    FakeDeviceMemory fake_dev_mem = 3;
    DeviceCambriconMemory device_cambricon_mem = 4;
  }
}
```

64. oneflow/core/memory/memory_case_util.h
```.h
#ifndef ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class MemCaseId {
 public:
  using device_index_t = MemZoneId::device_index_t;

  explicit MemCaseId(const MemoryCase& mem_case);
  explicit MemCaseId(const MemZoneId& mem_zone_id, DeviceType page_locked_device_type,
                     bool registered_by_network)
      : mem_zone_id_(mem_zone_id),
        host_mem_page_locked_device_type_(page_locked_device_type),
        host_mem_registered_by_network_(registered_by_network) {
    if (mem_zone_id.device_type() != DeviceType::kCPU) {
      CHECK_EQ(page_locked_device_type, DeviceType::kInvalidDevice);
      CHECK_EQ(registered_by_network, false);
    }
  }
  explicit MemCaseId(const MemZoneId& mem_zone_id, DeviceType page_locked_device_type)
      : MemCaseId(mem_zone_id, page_locked_device_type, false) {}
  explicit MemCaseId(const MemZoneId& mem_zone_id)
      : MemCaseId(mem_zone_id, DeviceType::kInvalidDevice, false) {}
  explicit MemCaseId(DeviceType device_type, device_index_t device_index,
                     DeviceType page_locked_device_type, bool registered_by_network)
      : MemCaseId(MemZoneId{device_type, device_index}, page_locked_device_type,
                  registered_by_network) {}
  explicit MemCaseId(DeviceType device_type, device_index_t device_index,
                     DeviceType page_locked_device_type)
      : MemCaseId(device_type, device_index, page_locked_device_type, false) {}
  explicit MemCaseId(DeviceType device_type, device_index_t device_index)
      : MemCaseId(device_type, device_index, DeviceType::kInvalidDevice, false) {}

  const MemZoneId& mem_zone_id() const { return mem_zone_id_; }
  DeviceType host_mem_page_locked_device_type() const { return host_mem_page_locked_device_type_; }
  bool is_host_mem_registered_by_network() const { return host_mem_registered_by_network_; }
  bool operator==(const MemCaseId& rhs) const {
    return mem_zone_id_ == rhs.mem_zone_id_
           && host_mem_page_locked_device_type_ == rhs.host_mem_page_locked_device_type_;
  }
  bool operator!=(const MemCaseId& rhs) const { return !((*this) == rhs); }

 private:
  MemZoneId mem_zone_id_;
  DeviceType host_mem_page_locked_device_type_;
  bool host_mem_registered_by_network_;
};

class GlobalMemCaseId {
 public:
  using rank_t = uint32_t;

  explicit GlobalMemCaseId(rank_t rank, const MemCaseId& mem_case_id)
      : rank_(rank), mem_case_id_(mem_case_id) {}
  explicit GlobalMemCaseId(rank_t rank, const MemoryCase& mem_case)
      : GlobalMemCaseId(rank, MemCaseId{mem_case}) {}

  rank_t rank() const { return rank_; }
  const MemCaseId& mem_case_id() const { return mem_case_id_; }
  bool operator==(const GlobalMemCaseId& rhs) const {
    return rank_ == rhs.rank_ && mem_case_id_ == rhs.mem_case_id_;
  }
  bool operator!=(const GlobalMemCaseId& rhs) const { return !((*this) == rhs); }

 private:
  rank_t rank_;
  MemCaseId mem_case_id_;
};

inline bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  return MemCaseId{lhs} == MemCaseId{rhs};
}

inline bool operator!=(const MemoryCase& lhs, const MemoryCase& rhs) {
  return !(MemCaseId{lhs} == MemCaseId{rhs});
}

int64_t SerializeMemCaseIdToInt64(const MemCaseId& mem_case_id);
void SerializeMemCaseIdToMemCase(const MemCaseId& mem_case_id, MemoryCase* mem_case);
int64_t SerializeGlobalMemCaseIdToInt64(const GlobalMemCaseId& mem_case_id);

bool PatchMemCaseId(MemCaseId* dst_mem_case_id, const MemCaseId& src_mem_case_id);
bool PatchMemCase(MemoryCase* dst_mem_case, const MemoryCase& src_mem_case);
MemCaseId GenerateCorrespondingPageLockedHostMemCaseId(const MemCaseId& mem_case_id);
MemoryCase GenerateCorrespondingPageLockedHostMemoryCase(const MemoryCase& mem_case);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_CASE_UTIL_H_

```

65. oneflow/core/memory/memory_case_util.cpp
```.cpp
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

namespace {

// MemCaseId int64_t encode
// |              | device_type | device_index  |                         |
// |              | ---- 5 ---- | ----- 7 ----- |                         |
// |              |         MemZoneId           |   pglck   | reg_by_net  |
// |              | ----------- 12 ------------ | --- 5 --- | ---- 1 ---- |
// |   reserved   |                       MemCaseId                       |
// | ---- 46 ---- | ------------------------ 18 ------------------------- |
// | ----------------------------- 64 bit ------------------------------- |

// GlobalMemCaseId int64_t encode
// |          |   rank   | MemCaseId  |
// |          | -- 19 -- | --- 18 --- |
// | reserved |    GlobalMemCaseId    |
// | -- 27 -- | -------- 37 --------- |
// | ------------ 64 bit ------------ |

constexpr size_t kRegByNetBits = 1;
constexpr size_t kPageLockedTypeBits = 5;
constexpr size_t kDeviceIndexBits = MemZoneId::kDeviceIndexBits;
constexpr size_t kDeviceTypeBits = MemZoneId::kDeviceTypeBits;

constexpr size_t kPageLockedTypeShift = kRegByNetBits;
constexpr size_t kDeviceIndexShift = kPageLockedTypeShift + kPageLockedTypeBits;
constexpr size_t kDeviceTypeShift = kDeviceIndexShift + kDeviceIndexBits;
constexpr size_t kRankShift = kDeviceTypeShift + kDeviceTypeBits;

}  // namespace

MemCaseId::MemCaseId(const MemoryCase& mem_case) {
  // TODO: consider migrate to registry
  DeviceType device_type = DeviceType::kInvalidDevice;
  device_index_t device_index = 0;
  DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
  bool host_mem_registered_by_network = false;
  if (mem_case.has_host_mem()) {
    device_type = DeviceType::kCPU;
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      page_locked_device_type = DeviceType::kGPU;
      device_index = mem_case.host_mem().cuda_pinned_mem().device_id();
    } else if (mem_case.host_mem().has_fake_dev_pinned_mem()) {
      page_locked_device_type = DeviceType::kFAKEDEVICE;
    } else {
      // host mem is pageable
    }
    if (mem_case.host_mem().has_used_by_network() && mem_case.host_mem().used_by_network()) {
      host_mem_registered_by_network = true;
    }
  } else if (mem_case.has_device_cuda_mem()) {
    device_type = DeviceType::kGPU;
    device_index = mem_case.device_cuda_mem().device_id();
  } else if (mem_case.has_fake_dev_mem()) {
    device_type = DeviceType::kFAKEDEVICE;
  } else if (mem_case.has_device_cambricon_mem()) {
    device_type = DeviceType::kCambricon;
    device_index = mem_case.device_cambricon_mem().device_id();
  } else {
    // Uninitialized MemoryCase, all member are set to default
  }
  mem_zone_id_ = MemZoneId{device_type, device_index};
  host_mem_page_locked_device_type_ = page_locked_device_type;
  host_mem_registered_by_network_ = host_mem_registered_by_network;
}

int64_t SerializeMemCaseIdToInt64(const MemCaseId& mem_case_id) {
  int64_t id = static_cast<int64_t>(mem_case_id.is_host_mem_registered_by_network());
  id |= static_cast<int64_t>(mem_case_id.host_mem_page_locked_device_type())
        << kPageLockedTypeShift;
  id |= static_cast<int64_t>(mem_case_id.mem_zone_id().device_index()) << kDeviceIndexShift;
  id |= static_cast<int64_t>(mem_case_id.mem_zone_id().device_type()) << kDeviceTypeShift;
  return id;
}

int64_t SerializeGlobalMemCaseIdToInt64(const GlobalMemCaseId& global_mem_case_id) {
  int64_t id = SerializeMemCaseIdToInt64(global_mem_case_id.mem_case_id());
  id |= static_cast<int64_t>(global_mem_case_id.rank()) << kRankShift;
  return id;
}

void SerializeMemCaseIdToMemCase(const MemCaseId& mem_case_id, MemoryCase* mem_case) {
  // TODO: consider migrate to registry
  if (mem_case_id.mem_zone_id().device_type() == DeviceType::kCPU) {
    auto* host_mem = mem_case->mutable_host_mem();
    if (mem_case_id.host_mem_page_locked_device_type() == DeviceType::kGPU) {
      host_mem->mutable_cuda_pinned_mem()->set_device_id(mem_case_id.mem_zone_id().device_index());
    } else if (mem_case_id.host_mem_page_locked_device_type() == DeviceType::kFAKEDEVICE) {
      host_mem->mutable_fake_dev_pinned_mem();
    } else {
      host_mem->Clear();
    }
    if (mem_case_id.is_host_mem_registered_by_network()) { host_mem->set_used_by_network(true); }
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(mem_case_id.mem_zone_id().device_index());
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kFAKEDEVICE) {
    mem_case->mutable_fake_dev_mem();
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kCambricon) {
    mem_case->mutable_device_cambricon_mem()->set_device_id(
        mem_case_id.mem_zone_id().device_index());
  } else {
    UNIMPLEMENTED();
  }
}

// Patch the source memory case to destination memory case.
// Patch failed when src_mem_case and dst_mem_case have different device_type
// or one of them has invalid device_type.
// Patch failed when src_mem_case and dst_mem_case have the same non-cpu device_type
// but have different device_index.
// When src_mem_case and dst_mem_case have the same cpu device_type
// and src_mem_case has more constrain than dst_mem_case(page-locked by other device,
// such as gpu or network device), patch the constrain of src_mem_case to dst_mem_case.
bool PatchMemCaseId(MemCaseId* dst_mem_case_id, const MemCaseId& src_mem_case_id) {
  DeviceType device_type = src_mem_case_id.mem_zone_id().device_type();
  if (device_type == DeviceType::kInvalidDevice) { return false; }
  if (device_type != dst_mem_case_id->mem_zone_id().device_type()) { return false; }

  if (device_type == DeviceType::kCPU) {
    MemCaseId::device_index_t device_index = dst_mem_case_id->mem_zone_id().device_index();
    auto page_locked_device_type = dst_mem_case_id->host_mem_page_locked_device_type();
    bool registered_by_network = dst_mem_case_id->is_host_mem_registered_by_network();
    if (src_mem_case_id.host_mem_page_locked_device_type() == DeviceType::kGPU) {
      page_locked_device_type = DeviceType::kGPU;
      device_index = src_mem_case_id.mem_zone_id().device_index();
    } else if (src_mem_case_id.host_mem_page_locked_device_type() == DeviceType::kFAKEDEVICE) {
      page_locked_device_type = DeviceType::kFAKEDEVICE;
    } else {
      // do nothing
    }
    if (src_mem_case_id.is_host_mem_registered_by_network()) { registered_by_network = true; }
    *dst_mem_case_id =
        MemCaseId{device_type, device_index, page_locked_device_type, registered_by_network};
  } else {
    if (dst_mem_case_id->mem_zone_id().device_index()
        != src_mem_case_id.mem_zone_id().device_index()) {
      return false;
    }
  }
  return true;
}

bool PatchMemCase(MemoryCase* dst_mem_case, const MemoryCase& src_mem_case) {
  MemCaseId src_mem_case_id{src_mem_case};
  MemCaseId dst_mem_case_id{*dst_mem_case};
  bool result = PatchMemCaseId(&dst_mem_case_id, src_mem_case_id);
  SerializeMemCaseIdToMemCase(dst_mem_case_id, dst_mem_case);
  return result;
}

MemCaseId GenerateCorrespondingPageLockedHostMemCaseId(const MemCaseId& mem_case_id) {
  CHECK_NE(mem_case_id.mem_zone_id().device_type(), DeviceType::kInvalidDevice);
  CHECK_NE(mem_case_id.mem_zone_id().device_type(), DeviceType::kCPU);
  DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
  MemCaseId::device_index_t device_index = 0;
  if (mem_case_id.mem_zone_id().device_type() == DeviceType::kGPU) {
    page_locked_device_type = DeviceType::kGPU;
    device_index = mem_case_id.mem_zone_id().device_index();
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kFAKEDEVICE) {
    page_locked_device_type = DeviceType::kFAKEDEVICE;
  } else {
    // do nothing
  }
  return MemCaseId{DeviceType::kCPU, device_index, page_locked_device_type};
}

MemoryCase GenerateCorrespondingPageLockedHostMemoryCase(const MemoryCase& mem_case) {
  MemCaseId host_mem_case_id = GenerateCorrespondingPageLockedHostMemCaseId(MemCaseId{mem_case});
  MemoryCase host_mem_case;
  SerializeMemCaseIdToMemCase(host_mem_case_id, &host_mem_case);
  return host_mem_case;
}

}  // namespace oneflow
```

66. oneflow/core/memory/memory_fake_dev_allocator.h
```.h
#ifndef ONEFLOW_CORE_MEMORY_MEMORY_FAKE_DEV_ALLOCATOR_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_FAKE_DEV_ALLOCATOR_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

struct FakeDevMemoryAllocatorImpl final {
  static void* Allocate(MemoryCase& mem_case, size_t size);
  static void Deallocate(void* ptr, MemoryCase mem_case);
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_FAKE_DEV_ALLOCATOR_H_
```

67. oneflow/core/memory/memory_fake_dev_allocator.cpp
```.cpp
#include "oneflow/core/memory/memory_fake_dev_allocator.h"
#include "oneflow/core/framework/device_register_fakedev.h"

namespace oneflow {
void* FakeDevMemoryAllocatorImpl::Allocate(MemoryCase& mem_case, size_t size) {
  void* ptr = nullptr;
  ptr = malloc(size + sizeof(FAKE_MAGIC_CODE));
  memcpy(ptr, &FAKE_MAGIC_CODE, sizeof(FAKE_MAGIC_CODE));
  CHECK_NOTNULL(ptr);
  return static_cast<char*>(ptr) + 4;
}

void FakeDevMemoryAllocatorImpl::Deallocate(void* ptr, MemoryCase mem_case) {
  free(static_cast<char*>(ptr) - 4);
}
}  // namespace oneflow

```

68. oneflow/core/register/register_manager.cpp
```.cpp
int64_t zone_id = SerializeMemCaseIdToInt64(MemCaseId{mem_block.mem_case()});
```

69. oneflow/core/register/runtime_register_desc.cpp
```.cpp
size_t RtRegstDesc::MainByteSize4OneRegst() const {
  if (!mem_case_.has_host_mem()) {
    return packed_blob_desc_->AlignedByteSizeOfBlobBody();
  } else {
    return packed_blob_desc_->AlignedTotalByteSize();
	@@ -94,7 +94,7 @@ size_t RtRegstDesc::TotalSeparatedHeaderByteSize4AllRegst() const {
}

size_t RtRegstDesc::SeparatedHeaderByteSize4OneRegst() const {
  if (!mem_case_.has_host_mem()) {
    return packed_blob_desc_->ByteSizeOfBlobHeader();
  } else {
    return 0;
```

70. oneflow/core/thread/cambricon_device_thread.h
```.h
#ifndef ONEFLOW_CORE_THREAD_CAMBRICON_DEVICE_THREAD_H_
#define ONEFLOW_CORE_THREAD_CAMBRICON_DEVICE_THREAD_H_

#ifdef WITH_CAMBRICON
#include "oneflow/core/thread/thread.h"

namespace oneflow {

class CambriconDeviceThread final : public Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CambriconDeviceThread);
  CambriconDeviceThread() = delete;
  ~CambriconDeviceThread();

  CambriconDeviceThread(int64_t thrd_id, int64_t dev_id);

 private:
  Channel<CambriconCBNotifier> cb_notifier_chan_;
  std::thread cb_notifier_poller_;
};

}  // namespace oneflow

#endif  // WITH_CAMBRICON
#endif  // ONEFLOW_CORE_THREAD_CAMBRICON_DEVICE_THREAD_H_

```

71. oneflow/core/thread/cambricon_device_thread.cpp
```.cpp
#ifdef WITH_CAMBRICON

#include "oneflow/core/thread/cambricon_device_thread.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/device/mlu_util.h"

namespace oneflow {

CambriconDeviceThread::CambriconDeviceThread(int64_t thrd_id, int64_t dev_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([=]() {
    MLUCurrentDeviceGuard guard(dev_id);
    ThreadCtx thread_ctx;
    thread_ctx.g_cambricon_queue.reset(new CambriconQueueHandle(&cb_notifier_chan_));
    thread_ctx.cb_notifier_chan = &cb_notifier_chan_;
    PollMsgChannel(thread_ctx);
  });

  cb_notifier_poller_ = std::thread([=]() {
    MLUCurrentDeviceGuard guard(dev_id);
    CambriconCBNotifier cb_notifier;
    while (cb_notifier_chan_.Receive(&cb_notifier) == kChannelStatusSuccess) {
      CNRT_CHECK(cnrtWaitNotifier(cb_notifier.notifier));
      cb_notifier.callback();
      CNRT_CHECK(cnrtDestroyNotifier(&cb_notifier.notifier));
    }
  });
}

CambriconDeviceThread::~CambriconDeviceThread() {
  cb_notifier_chan_.Close();
  cb_notifier_poller_.join();
}

REGISTER_DEVICE_THREAD_CREATOR_WITH_STREAM_ID(
    DeviceType::kCambricon, ([](const StreamId& stream_id) -> Thread* {
      int64_t thrd_id = SerializeStreamIdToInt64(stream_id);
      int64_t dev_id = static_cast<int64_t>(stream_id.device_id().device_index());
      return new CambriconDeviceThread(thrd_id, dev_id);
    }));

}  // namespace oneflow

#endif  // WITH_CAMBRICON
```

72. oneflow/core/thread/fake_device_thread.h
```.h
#ifndef ONEFLOW_CORE_THREAD_FAKE_DEVICE_THREAD_H_
#define ONEFLOW_CORE_THREAD_FAKE_DEVICE_THREAD_H_

#include "oneflow/core/thread/thread.h"

namespace oneflow {

class FakeDeviceThread final : public Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FakeDeviceThread);
  FakeDeviceThread() = delete;
  ~FakeDeviceThread() = default;

  FakeDeviceThread(int64_t thrd_id);

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_FAKE_DEVICE_THREAD_H_
```

73. oneflow/core/thread/fake_device_thread.cpp
```.cpp
#include "oneflow/core/thread/fake_device_thread.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/graph/id_serialization.h"

namespace oneflow {

FakeDeviceThread::FakeDeviceThread(int64_t thrd_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, thrd_id]() {
    OF_PROFILER_NAME_THIS_HOST_THREAD("Fake Device Actor : (" + std::to_string(thrd_id) + ")");
    ThreadCtx ctx;
#ifdef WITH_CUDA
    ctx.cb_event_chan = nullptr;
#endif  // WITH_CUDA
    PollMsgChannel(ctx);
  });
}

REGISTER_DEVICE_THREAD_CREATOR_WITH_STREAM_ID(DeviceType::kFAKEDEVICE,
                                              ([](const StreamId& stream_id) -> Thread* {
                                                return new FakeDeviceThread(
                                                    SerializeStreamIdToInt64(stream_id));
                                              }));

}  // namespace oneflow
```

74. oneflow/core/thread/thread_context.h
```.h
#include "oneflow/core/device/cambricon_queue_handle.h"


#ifdef WITH_CAMBRICON
  std::unique_ptr<CambriconQueueHandle> g_cambricon_queue;
  Channel<CambriconCBNotifier>* cb_notifier_chan;
#endif
```

75. oneflow/core/thread/thread_manager.cpp
```.cpp
#include "oneflow/core/thread/fake_device_thread.h"
```



76. oneflow/python/experimental/interface_op_read_and_write.py 
```.py
def GetEagerInterfaceBlob(op_name):
    flow.sync_default_session()
    sess = session_ctx.GetDefaultSession()

    def CreateBlob():
```

77. oneflow/python/framework/config_util.py 
```.py
@oneflow_export("config.mlu_device_num")
def api_mlu_device_num(val: int) -> None:
    r"""Set number of MLUs on each machine to run oneflow on. Usually you don't need to set this.
    Args:
        val (int): number of MLUs. It is identical on every machine.
    """
    return enable_if.unique([mlu_device_num, do_nothing])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.session_initialized)
def mlu_device_num(val):
    sess = session_ctx.GetDefaultSession()
    assert type(val) is int
    sess.config_proto.resource.mlu_device_num = val
```

78. oneflow/python/framework/device.py 
```.py
import oneflow.core.common.device_type_pb2 as device_type_pb
from oneflow.python.oneflow_export import oneflow_export


_device_tag_2_device_type = {
    "cpu": device_type_pb.kCPU,
    "gpu": device_type_pb.kGPU,
    "dummy": device_type_pb.kFAKEDEVICE,
    "cambricon": device_type_pb.kCambricon,
}


_device_type_2_device_tag = {
    device_type_pb.kCPU: "cpu",
    device_type_pb.kGPU: "gpu",
    device_type_pb.kFAKEDEVICE: "dummy",
    device_type_pb.kCambricon: "cambricon",
}


@oneflow_export("is_valid_device_tag")
def is_valid_device_tag(device_tag: str):
    if not isinstance(device_tag, str):
        return False
    return device_tag in _device_tag_2_device_type
```

79. oneflow/python/framework/placement_context.py
```.py
def GetMluMachineDeviceIds(resource):
    assert resource.machine_num > 0
    assert resource.HasField("mlu_device_num")
    return [
        "{}:0-{}".format(m_id, resource.mlu_device_num - 1)
        for m_id in range(resource.machine_num)
    ]
```

80. oneflow/python/framework/placement_util.py
```.py
elif resource.HasField("mlu_device_num"):
        return "cambricon", placement_ctx.GetMluMachineDeviceIds(resource)
```

81. oneflow/python/framework/session_util.py
```.py
def _GetDefaultConfigProto():
    config_proto = job_set_util.ConfigProto()
    config_proto.resource.machine_num = 0
    if oneflow._oneflow_internal.flags.with_cuda():
        config_proto.resource.gpu_device_num = 1
    else:
        config_proto.resource.cpu_device_num = 1
        config_proto.resource.gpu_device_num = 0
    config_proto.io_conf.SetInParent()
    config_proto.session_id = session_ctx.GetDefaultSession().id
    return config_proto
```

82. oneflow/python/ops/layers.py 
```.py
data_format = data_format.upper()

    if data_format != "NCHW" and data_format != "NHWC":
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    need_transpose = 0
    if flow.current_scope().device_parallel_desc_symbol.device_tag == "cambricon":
        if data_format == "NCHW":
            need_transpose = 1
        data_format = "channels_last"
    else:
        if data_format == "NHWC":
            need_transpose = 1
        data_format = "channels_first"

    if need_transpose:
        x = flow.transpose(x, perm=[0, 3, 1, 2])
	@@ -1543,7 +1551,7 @@ def upsample_Job(x: tp.Numpy.Placeholder((1, 32, 32, 32))
        .Output("y")
        .Attr("height_scale", float(height_scale))
        .Attr("width_scale", float(width_scale))
        .Attr("data_format", data_format)
        .Attr("interpolation", interpolation)
        .Build()
    )
```

83. oneflow/python/ops/nn_ops.py
```.py
if flow.current_scope().device_parallel_desc_symbol.device_tag == "cambricon":
	channel = x.shape[3]
	gamma = flow.get_variable(
			name + "_gamma",
			shape=(channel,),
			dtype=x.dtype,
			initializer=flow.ones_initializer(),
			trainable=False,
	)
	beta = flow.get_variable(
			name + "_beta",
			shape=(channel,),
			dtype=x.dtype,
			initializer=flow.zeros_initializer(),
			trainable=False,
	)
	out, mean, var = (
			flow.user_op_builder(name)
			.Op("instance_norm_2d")
			.Input("in", [x])
			.Input("gamma", [gamma])
			.Input("beta", [beta])
			.Output("out")
			.Output("mean")
			.Output("var")
			.Attr("eps", eps)
			.Build()
			.InferAndTryRun()
			.RemoteBlobList()
	)
	return out, mean, var
```

84. oneflow/python/serving/inference_session.py 
```.py
def _make_config_proto(self):
	if self.config_proto_ is None:
			self.config_proto_ = session_util._GetDefaultConfigProto()
			self.config_proto_.resource.ClearField("cpu_device_num")
			self.config_proto_.resource.ClearField("gpu_device_num")
			self.config_proto_.resource.ClearField("mlu_device_num")

	if self.option_.device_tag == "cpu":
			self.config_proto_.resource.cpu_device_num = self.option_.device_num
	elif self.option_.device_tag == "gpu":
			self.config_proto_.resource.gpu_device_num = self.option_.device_num
	elif self.option_.device_tag == "cambricon":
			self.config_proto_.resource.mlu_device_num = self.option_.device_num
	else:
			raise NotImplementedError(
					"not supported device tag {}".format(self.option_.device_tag)
	@@ -291,6 +295,7 @@ def compile(self, op_list):
									op_conf.name, op_conf.device_tag, device_tag
							)
					)
					op_conf.device_tag = device_tag

			compile_ctx.CurJobAddOp(op_conf)
```

85. oneflow/python/serving/saved_model_builder.py
```.py
# flow.checkpoint.save(checkpoint_path)
        # using old api to save checkpoint for temporarily because eager api don't support new device type
        check_point = flow.train.CheckPoint()
        check_point.save(checkpoint_path)
```

86. oneflow/python/test/models/insightface/fresnet100.py
```.py
import oneflow as flow

def _get_initializer():
    return flow.random_normal_initializer(mean=0.0, stddev=0.1)

def _get_regularizer(name):
    return None

def _prelu(inputs, data_format="NCHW", name=None):
    return flow.layers.prelu(
        inputs,
        alpha_initializer=flow.constant_initializer(0.25),
        alpha_regularizer=_get_regularizer("alpha"),
        shared_axes=[2, 3] if data_format == "NCHW" else [1, 2],
        name=name,
    )


def _batch_norm(
    inputs,
    epsilon,
    center=True,
    scale=True,
    is_training=True,
    data_format="NCHW",
    name=None,
):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=3 if data_format == "NHWC" and len(inputs.shape) == 4 else 1,
        momentum=0.9,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=flow.zeros_initializer(),
        gamma_initializer=flow.ones_initializer(),
        beta_regularizer=None,
        gamma_regularizer=None,
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=is_training,
        training=is_training,
        name=name,
    )

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num=1,
    data_format="NCHW",
    dilation_rate=1,
    activation=None,
    use_bias=False,
    weight_initializer=_get_initializer(),
    bias_initializer=flow.zeros_initializer(),
    weight_regularizer=_get_regularizer("weight"),
    bias_regularizer=_get_regularizer("bias"),
):
    return flow.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, groups=group_num, activation=activation, use_bias=use_bias, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer, name=name)

def residual_unit_v3(
    in_data, num_filter, stride, dim_match, bn_is_training, data_format, name
):

    suffix = ""
    use_se = 0
    bn1 = _batch_norm(
        in_data,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_bn1" % (name, suffix),
    )
    conv1 = _conv2d_layer(
        name="%s%s_conv1" % (name, suffix),
        input=bn1,
        filters=num_filter,
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn2 = _batch_norm(
        conv1,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_bn2" % (name, suffix),
    )
    prelu = _prelu(bn2, data_format=data_format, name="%s%s_relu1" % (name, suffix))
    conv2 = _conv2d_layer(
        name="%s%s_conv2" % (name, suffix),
        input=prelu,
        filters=num_filter,
        kernel_size=3,
        strides=stride,
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn3 = _batch_norm(
        conv2,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_bn3" % (name, suffix),
    )

    if dim_match:
        input_blob = in_data
    else:
        input_blob = _conv2d_layer(
            name="%s%s_conv1sc" % (name, suffix),
            input=in_data,
            filters=num_filter,
            kernel_size=1,
            strides=stride,
            padding="valid",
            data_format=data_format,
            use_bias=False,
            dilation_rate=1,
            activation=None,
        )
        input_blob = _batch_norm(
            input_blob,
            epsilon=2e-5,
            is_training=bn_is_training,
            data_format=data_format,
            name="%s%s_sc" % (name, suffix),
        )

    identity = flow.math.add(x=bn3, y=input_blob)
    return identity


def get_symbol(input_blob):
    filter_list = [64, 64, 128, 256, 512]
    num_stages = 4
    units = [3, 13, 30, 3]
    num_classes = 512
    bn_is_training = False
    data_format = "NHWC"
    if data_format.upper() == "NCHW":
        input_blob = flow.transpose(
        input_blob, name="transpose", perm=[0, 3, 1, 2]
    )
    input_blob = _conv2d_layer(
        name="conv0",
        input=input_blob,
        filters=filter_list[0],
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    input_blob = _batch_norm(
        input_blob, epsilon=2e-5, is_training=bn_is_training, data_format=data_format, name="bn0"
    )
    input_blob = _prelu(input_blob, data_format=data_format, name="relu0")

    for i in range(num_stages):
        input_blob = residual_unit_v3(
            input_blob,
            filter_list[i + 1],
            [2, 2],
            False,
            bn_is_training=bn_is_training,
            data_format=data_format,
            name="stage%d_unit%d" % (i + 1, 1),
        )
        for j in range(units[i] - 1):
            input_blob = residual_unit_v3(
                input_blob,
                filter_list[i + 1],
                [1, 1],
                True,
                bn_is_training=bn_is_training,
                data_format=data_format,
                name="stage%d_unit%d" % (i + 1, j + 2),
            )
    body = _batch_norm(
            input_blob,
            epsilon=2e-5,
            is_training=False,
            data_format="NHWC",
            name="bn1"
    )
    body = flow.reshape(body, (body.shape[0], -1))
    pre_fc1 = flow.layers.dense(
            inputs=body,
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            trainable=True,
            name="pre_fc1",
    )
    fc1 = _batch_norm(
            pre_fc1,
            epsilon=2e-5,
            scale=True,
            center=True,
            is_training=False,
            data_format="NHWC",
            name="fc1",
    )
    return fc1

```

87. oneflow/python/test/models/insightface/insightface_val.py
```.py
import math
import os
import argparse
import numpy as np
import cv2
import oneflow as flow

import fresnet100
import oneflow.typing as tp
from typing import Tuple
from scipy.spatial import distance

def get_val_args():
    val_parser = argparse.ArgumentParser(description="flags for validation")
    val_parser.add_argument(
            "--val_img_dir",
            type=str,
            default="./woman.jpeg",
            help="validation dataset dir",
        )

    # distribution config
    val_parser.add_argument(
        "--device_num_per_node",
        type=int,
        default=1,
        required=False,
    )
    val_parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="node/machine number for training",
    )

    val_parser.add_argument(
        "--val_batch_size",
        default=1,
        type=int,
        help="validation batch size totally",
    )
    # model and log
    val_parser.add_argument(
        "--log_dir", type=str, default="./log", help="log info save"
    )
    val_parser.add_argument(
        "--model_load_dir", default="/insightface_nhwc", help="path to load model."
    )
    return val_parser.parse_args()


def load_image(image_path):
    im = cv2.imread(image_path)
    dsize = (112, 112)
    rgb_mean = [127.5, 127.5, 127.5]
    std_values = [128.0, 128.0, 128.0]

    im = cv2.resize(im, dsize, interpolation = cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = (im - rgb_mean) / std_values
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 2, 3, 1))
    print("image size: ", im.shape)
    return np.ascontiguousarray(im, 'float32')

def get_cambricon_config():
    val_config = flow.function_config()
    val_config.default_logical_view(flow.scope.consistent_view())
    val_config.default_data_type(flow.float)
    val_config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))
    return val_config

def validation_job(images, config):
    @flow.global_function(type="predict", function_config=config)
    def get_symbol_val_job(
            images: flow.typing.Numpy.Placeholder(
                (1, 112, 112, 3)
            )
        ):
        print("val batch data: ", images.shape)
        embedding = fresnet100.get_symbol(images)
        return embedding

    return get_symbol_val_job

def do_validation(images, val_job, name_suffix):
    print("Validation starts...")
    batch_size = 1
    total_images_num = 1

    _em = val_job(images).get()
    return _em


def load_checkpoint(model_load_dir):
    print("=" * 20 + " model load begin " + "=" * 20)
    flow.train.CheckPoint().load(model_load_dir)
    print("=" * 20 + " model load end " + "=" * 20)


def main():
    args = get_val_args()
    flow.env.init()
    flow.env.log_dir(args.log_dir)
    # validation
    print("args: ", args)
    output_list = [] 
    if os.path.exists(args.val_img_dir):
        print("=" * 20 + " image load begin " + "=" * 20)
        images = load_image(args.val_img_dir)
        print("=" * 20 + " image load end " + "=" * 20)
    else: 
        raise ValueError ("Image path for validation does NOT exist!")
    flow.config.enable_legacy_model_io(True)
    val_job = validation_job(images, get_cambricon_config())
    load_checkpoint(args.model_load_dir)
    print("=" * 20 + " Prediction begins " + "=" * 20)   
    mlu_res = do_validation(images, val_job, "mlu")
    print("=" * 20 + " Prediction ends " + "=" * 20)
    flow.clear_default_session()

if __name__ == "__main__":
    main()
```

88. oneflow/python/test/models/insightface/save_insightface_model.py 
```.py
"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import shutil
import argparse
import oneflow as flow
import fresnet100


def _init_oneflow_env_and_config():
    flow.env.init()
    flow.enable_eager_execution(False)
    flow.config.enable_legacy_model_io(True)


def _make_insightface_predict_func(width, height):
    batch_size = 1
    channels = 3

    func_cfg = flow.function_config()
    func_cfg.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

    @flow.global_function("predict", function_config=func_cfg)
    def predict_fn(
        image: flow.typing.Numpy.Placeholder(
            shape=(batch_size, height, width, channels), dtype=flow.float32
        )
    ) -> flow.typing.Numpy:
        embeding = fresnet100.get_symbol(image)
        return embeding

    return predict_fn


def main(args):
    _init_oneflow_env_and_config()

    predict_fn = _make_insightface_predict_func(args.image_width, args.image_height)
    flow.train.CheckPoint().load(args.model_dir)
    # flow.load_variables(flow.checkpoint.get(args.model_dir))
    print("predict_fn construct finished")

    saved_model_path = args.save_dir
    model_version = args.model_version

    model_version_path = os.path.join(saved_model_path, str(model_version))
    if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
        if args.force_save:
            print(
                f"WARNING: The model version path '{model_version_path}' already exist"
                ", old version directory will be replaced"
            )
            shutil.rmtree(model_version_path)
        else:
            raise ValueError(
                f"The model version path '{model_version_path}' already exist"
            )

    saved_model_builder = (
        flow.saved_model.ModelBuilder(saved_model_path)
        .ModelName(args.model_name)
        .Version(model_version)
    )
    saved_model_builder.AddFunction(predict_fn).Finish()
    saved_model_builder.Save()


def _parse_args():
    parser = argparse.ArgumentParser("flags for save insightface model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="stylenet_nhwc",
        help="model parameters directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="insightface_models",
        help="directory to save models",
    )
    parser.add_argument(
        "--model_name", type=str, default="insightface", help="model name"
    )
    parser.add_argument("--model_version", type=int, default=1, help="model version")
    parser.add_argument(
        "--force_save",
        default=False,
        action="store_true",
        help="force save model whether already exists or not",
    )
    parser.add_argument(
        "--image_width", type=int, default=224, help="input image width"
    )
    parser.add_argument(
        "--image_height", type=int, default=224, help="input image height"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
```

89. oneflow/python/test/models/insightface/save_insightface_model.sh
```.sh
#!/bin/bash
set -ex

# download model parameters for first-time 
# wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/insightface.tar.gz
# tar zxvf insightface.tar.gz

base_dir=`dirname $0`

python3 $base_dir/save_insightface_model.py \
    --model_dir insightface \
    --save_dir insightface_models \
    --model_version 1 \
    --image_width 112 \
    --image_height 112 \
    --force_save
```

90. oneflow/python/test/models/insightface/validate.sh
```.sh
# !/bin/bash

MODEL_LOAD_DIR="./insightface_nhwc/"

INPUT_IMAGE="./images/dog.jpeg"

python3 insightface_val.py \
    --val_img_dir $INPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR
```

91. oneflow/python/test/models/resnet50/README.md 
```.md
# resnet50 classification

## 使用方法

- 从`https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/resnet50.tar.gz`下载模型和图片到当前`infer.sh`所在的目录
- 执行`sh infer.sh`即可获得 resnet50 分类预测结果
```

92. oneflow/python/test/models/resnet50/imagenet1000_clsidx_to_labels.py
```.py
clsidx_2_labels = {
    0: "tench, Tinca tinca",
    1: "goldfish, Carassius auratus",
    2: "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",

```

93. oneflow/python/test/models/resnet50/infer.sh
```.sh
MODEL_LOAD_DIR="./resnet50_nhwc/"

INPUT_IMAGE="./images/fish.jpg"

python3 infer_resnet50.py \
    --input_image_path $INPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR
```

94. oneflow/python/test/models/resnet50/infer_resnet50.py
```.py
import oneflow as flow
import oneflow.typing as tp

import argparse
import cv2
import numpy as np

from imagenet1000_clsidx_to_labels import clsidx_2_labels
from resnet50_model import resnet50

def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    im = (im - [123.68, 116.779, 103.939]) / [58.393, 57.12, 57.375]
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


flow.config.enable_legacy_model_io(True)


def main(args):
    input_image = load_image(args.input_image_path)
    height = 224
    width = 224
    flow.env.init()
    config = flow.function_config()
    config.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

    @flow.global_function("predict", function_config=config)
    def InferenceNet(
        images: tp.Numpy.Placeholder((1, height, width, 3), dtype=flow.float)
    ) -> tp.Numpy:
        logits = resnet50(images, args, training=False)
        predictions = flow.nn.softmax(logits)
        return predictions

    print("===============================>load begin")
    flow.train.CheckPoint().load(args.model_load_dir)
    print("===============================>load end")

    import datetime

    a = datetime.datetime.now()

    print("predict begin")
    reset_out = InferenceNet(input_image)
    print("predict end")
    clsidx = reset_out.argmax()

    b = datetime.datetime.now()
    c = b - a

    print("time: %s ms, height: %d, width: %d" % (c.microseconds / 1000, height, width))
    print(
        "resnet50 predict prob %f, class %s"
        % (reset_out.max(), clsidx_2_labels[clsidx])
    )


def get_parser(parser=None):
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("flags for neural style")
    parser.add_argument(
        "--input_image_path", type=str, default="images/tiger.jpg", help="image path"
    )
    parser.add_argument(
        "--model_load_dir", type=str, default="", help="model save directory"
    )
    parser.add_argument(
        "--channel_last",
        type=str2bool,
        default=True,
        help="Whether to use use channel last mode(nhwc)",
    )
    # fuse bn relu or bn add relu
    parser.add_argument(
        "--fuse_bn_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--fuse_bn_add_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization add relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--pad_output",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to pad the output to number of image channels to 4.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
```

95. oneflow/python/test/models/resnet50/resnet50_model.py
```.py
import oneflow as flow

BLOCK_COUNTS = [3, 4, 6, 3]
BLOCK_FILTERS = [256, 512, 1024, 2048]
BLOCK_FILTERS_INNER = [64, 128, 256, 512]


class ResnetBuilder(object):
    def __init__(
        self,
        weight_regularizer,
        trainable=True,
        training=True,
        channel_last=False,
        fuse_bn_relu=True,
        fuse_bn_add_relu=True,
    ):
        self.data_format = "NHWC" if channel_last else "NCHW"
        self.weight_initializer = flow.variance_scaling_initializer(
            2, "fan_in", "random_normal", data_format=self.data_format
        )
        self.weight_regularizer = weight_regularizer
        self.trainable = trainable
        self.training = training
        self.fuse_bn_relu = fuse_bn_relu
        self.fuse_bn_add_relu = fuse_bn_add_relu

    def _conv2d(
        self, name, input, filters, kernel_size, strides=1, padding="SAME", dilations=1,
    ):
        # There are different shapes of weight metric between 'NCHW' and 'NHWC' mode
        if self.data_format == "NHWC":
            shape = (filters, kernel_size, kernel_size, input.shape[3])
        else:
            shape = (filters, input.shape[1], kernel_size, kernel_size)
        weight = flow.get_variable(
            name + "-weight",
            shape=shape,
            dtype=input.dtype,
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            model_name="weight",
            trainable=self.trainable,
        )

        return flow.nn.conv2d(
            input, weight, strides, padding, self.data_format, dilations, name=name
        )

    def _batch_norm(self, inputs, name=None, last=False):
        initializer = flow.zeros_initializer() if last else flow.ones_initializer()
        axis = 1
        if self.data_format == "NHWC":
            axis = 3
        return flow.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            momentum=0.9,  # 97,
            epsilon=1e-5,
            center=True,
            scale=True,
            trainable=self.trainable,
            training=self.training,
            gamma_initializer=initializer,
            moving_variance_initializer=initializer,
            gamma_regularizer=self.weight_regularizer,
            beta_regularizer=self.weight_regularizer,
            name=name,
        )

    def _batch_norm_relu(self, inputs, name=None, last=False):
        if self.fuse_bn_relu:
            initializer = flow.zeros_initializer() if last else flow.ones_initializer()
            axis = 1
            if self.data_format == "NHWC":
                axis = 3
            return flow.layers.batch_normalization_relu(
                inputs=inputs,
                axis=axis,
                momentum=0.9,
                epsilon=1e-5,
                center=True,
                scale=True,
                trainable=self.trainable,
                training=self.training,
                gamma_initializer=initializer,
                moving_variance_initializer=initializer,
                gamma_regularizer=self.weight_regularizer,
                beta_regularizer=self.weight_regularizer,
                name=name + "_bn_relu",
            )
        else:
            return flow.nn.relu(self._batch_norm(inputs, name + "_bn", last=last))

    def _batch_norm_add_relu(self, inputs, addend, name=None, last=False):
        if self.fuse_bn_add_relu:
            initializer = flow.zeros_initializer() if last else flow.ones_initializer()
            axis = 1
            if self.data_format == "NHWC":
                axis = 3
            return flow.layers.batch_normalization_add_relu(
                inputs=inputs,
                addend=addend,
                axis=axis,
                momentum=0.9,
                epsilon=1e-5,
                center=True,
                scale=True,
                trainable=self.trainable,
                training=self.training,
                gamma_initializer=initializer,
                moving_variance_initializer=initializer,
                gamma_regularizer=self.weight_regularizer,
                beta_regularizer=self.weight_regularizer,
                name=name + "_bn_add_relu",
            )
        else:
            return flow.nn.relu(
                self._batch_norm(inputs, name + "_bn", last=last) + addend
            )

    def conv2d_affine(self, input, name, filters, kernel_size, strides):
        # input data_format must be NCHW, cannot check now
        padding = "SAME" if strides > 1 or kernel_size > 1 else "VALID"
        output = self._conv2d(name, input, filters, kernel_size, strides, padding)
        return output

    def bottleneck_transformation(
        self, input, block_name, filters, filters_inner, strides
    ):
        a = self.conv2d_affine(input, block_name + "_branch2a", filters_inner, 1, 1)
        a = self._batch_norm_relu(a, block_name + "_branch2a")

        b = self.conv2d_affine(a, block_name + "_branch2b", filters_inner, 3, strides)
        b = self._batch_norm_relu(b, block_name + "_branch2b")

        c = self.conv2d_affine(b, block_name + "_branch2c", filters, 1, 1)
        return c

    def residual_block(self, input, block_name, filters, filters_inner, strides_init):
        if strides_init != 1 or block_name == "res2_0":
            shortcut = self.conv2d_affine(
                input, block_name + "_branch1", filters, 1, strides_init
            )
            shortcut = self._batch_norm(shortcut, block_name + "_branch1_bn")
        else:
            shortcut = input

        bottleneck = self.bottleneck_transformation(
            input, block_name, filters, filters_inner, strides_init,
        )
        return self._batch_norm_add_relu(
            bottleneck, shortcut, block_name + "_branch2c", last=True
        )

    def residual_stage(
        self, input, stage_name, counts, filters, filters_inner, stride_init=2
    ):
        output = input
        for i in range(counts):
            block_name = "%s_%d" % (stage_name, i)
            output = self.residual_block(
                output, block_name, filters, filters_inner, stride_init if i == 0 else 1
            )

        return output

    def resnet_conv_x_body(self, input):
        output = input
        for i, (counts, filters, filters_inner) in enumerate(
            zip(BLOCK_COUNTS, BLOCK_FILTERS, BLOCK_FILTERS_INNER)
        ):
            stage_name = "res%d" % (i + 2)
            output = self.residual_stage(
                output, stage_name, counts, filters, filters_inner, 1 if i == 0 else 2
            )
        return output

    def resnet_stem(self, input):
        conv1 = self._conv2d("conv1", input, 64, 7, 2)
        conv1_bn = self._batch_norm_relu(conv1, "conv1")
        pool1 = flow.nn.max_pool2d(
            conv1_bn,
            ksize=3,
            strides=2,
            padding="SAME",
            data_format=self.data_format,
            name="pool1",
        )
        return pool1


def resnet50(images, args, trainable=True, training=True):
    weight_regularizer = None
    builder = ResnetBuilder(
        weight_regularizer,
        trainable,
        training,
        args.channel_last,
        args.fuse_bn_relu,
        args.fuse_bn_add_relu,
    )
    if args.pad_output:
        if args.channel_last:
            paddings = ((0, 0), (0, 0), (0, 0), (0, 1))
        else:
            paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        images = flow.pad(images, paddings=paddings)
    with flow.scope.namespace("Resnet"):
        stem = builder.resnet_stem(images)
        body = builder.resnet_conv_x_body(stem)
        pool5 = flow.nn.avg_pool2d(
            body,
            ksize=7,
            strides=1,
            padding="VALID",
            data_format=builder.data_format,
            name="pool5",
        )
        fc1001 = flow.layers.dense(
            flow.reshape(pool5, (pool5.shape[0], -1)),
            units=1000,
            use_bias=True,
            kernel_initializer=flow.variance_scaling_initializer(
                2, "fan_in", "random_normal"
            ),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=weight_regularizer,
            bias_regularizer=weight_regularizer,
            trainable=trainable,
            name="fc1001",
        )
    return fc1001
```

96. oneflow/python/test/models/resnet50/save_model.py
```.py
import argparse
import os
import shutil

import oneflow as flow
import oneflow.typing as tp

from resnet50_model import resnet50

def _init_oneflow_env_and_config():
    flow.env.init()
    flow.enable_eager_execution(False)
    flow.config.enable_legacy_model_io(True)

def _make_resnet50_predict_func(args):
    batch_size = 1
    channels = 3

    func_cfg = flow.function_config()
    func_cfg.default_placement_scope(flow.scope.placement("cambricon", "0:0"))

    @flow.global_function("predict", function_config=func_cfg)
    def predict_fn(
        images: tp.Numpy.Placeholder((1, args.image_height, args.image_width, channels), dtype=flow.float)
    ) -> tp.Numpy:
        logits = resnet50(images, args, training=False)
        predictions = flow.nn.softmax(logits)
        return predictions

    return predict_fn


def main(args):
    _init_oneflow_env_and_config()

    predict_fn = _make_resnet50_predict_func(args)
    flow.train.CheckPoint().load(args.model_dir)
    print("predict_fn construct finished")

    saved_model_path = args.save_dir
    model_version = args.model_version

    model_version_path = os.path.join(saved_model_path, str(model_version))
    if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
        if args.force_save:
            print(
                f"WARNING: The model version path '{model_version_path}' already exist"
                ", old version directory will be replaced"
            )
            shutil.rmtree(model_version_path)
        else:
            raise ValueError(
                f"The model version path '{model_version_path}' already exist"
            )

    saved_model_builder = (
        flow.saved_model.ModelBuilder(saved_model_path)
        .ModelName(args.model_name)
        .Version(model_version)
    )
    saved_model_builder.AddFunction(predict_fn).Finish()
    saved_model_builder.Save()


def _parse_args():
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("flags for save resnet50 model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="resnet50_nhwc",
        help="model parameters directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="resnet50_models",
        help="directory to save models",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet50", help="model name"
    )
    parser.add_argument("--model_version", type=int, default=1, help="model version")
    parser.add_argument(
        "--force_save",
        default=False,
        action="store_true",
        help="force save model whether already exists or not",
    )
    parser.add_argument(
        "--image_width", type=int, default=224, help="input image width"
    )
    parser.add_argument(
        "--image_height", type=int, default=224, help="input image height"
    )
    parser.add_argument(
        "--channel_last",
        type=str2bool,
        default=True,
        help="Whether to use use channel last mode(nhwc)",
    )
    # fuse bn relu or bn add relu
    parser.add_argument(
        "--fuse_bn_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--fuse_bn_add_relu",
        type=str2bool,
        default=False,
        help="Whether to use use fuse batch normalization add relu. Currently supported in origin/master of OneFlow only.",
    )
    parser.add_argument(
        "--pad_output",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to pad the output to number of image channels to 4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
```

97. oneflow/python/test/models/resnet50/save_model.sh
```.sh
#!/bin/bash
set -ex

# download model parameters for first-time 
# wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/resnet50.tar.gz
# tar zxvf resnet50.tar.gz

base_dir=`dirname $0`

python3 $base_dir/save_model.py \
    --model_dir resnet50_nhwc \
    --save_dir resnet50_models \
    --model_version 1 \
    --force_save
```

98. oneflow/python/test/models/styletransform/README.md
```.md
# styletransform

## 使用方法

- 从`https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/styletranform.tar.gz`下载模型和图片到当前`infer.sh`所在的目录
- 执行`sh infer.sh`即可获得StyleNet风格化（素描）后的结果图片

## 保存 model（for serving）

首先下载并解压模型所需的参数

`bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/styletranform.tar.gz
tar zxvf styletranform.tar.gz
`

如果想导出 gpu 平台的模型，则要在支持 gpu 的环境中，执行
`bash
bash save_style_model.sh gpu
`

如果想导出 寒武纪 平台的模型，则要在支持 寒武纪 的环境中，执行
`bash
bash save_style_model.sh cambricon
`

save_style_model.sh 中的参数
- backend: 设置 device 类型
- model_dir: 模型所需的参数的目录，通过上面的下载解压命令后可以得到的 stylenet_nhwc 目录即是参数目录
- save_dir: 模型保存的目录
- model_version: 保存模型的版本号
- image_width: 模型兼容的输入 image 的 width
- image_height: 模型兼容的输入 image 的 height
- force_save: 当 model_version 所指定的保存的模型的版本号已存在时，是否强制覆盖保存
```

99. oneflow/python/test/models/styletransform/infer.sh
```.sh
MODEL_LOAD_DIR="./stylenet_nhwc/"

INPUT_IMAGE="./images/content-images/amber.jpg"
OUTPUT_IMAGE="./images/style_out_amber_nhwc.jpg"

BACKEND=${1:-gpu}

python3 infer_of_neural_style.py \
    --backend $BACKEND \
    --input_image_path $INPUT_IMAGE \
    --output_image_path $OUTPUT_IMAGE \
    --model_load_dir $MODEL_LOAD_DIR
```

100. oneflow/python/test/models/styletransform/infer_of_neural_style.py
```.py
import numpy as np
import argparse
import cv2

import oneflow as flow
import oneflow.typing as tp
import style_model


def float_list(x):
    return list(map(float, x.split(",")))


def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 2, 3, 1))
    return np.ascontiguousarray(im, "float32")


def recover_image(im):
    im = np.squeeze(im)
    print(im.shape)
    # im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)


flow.config.enable_legacy_model_io(True)


def main(args):
    input_image = load_image(args.input_image_path)
    height = input_image.shape[1]
    width = input_image.shape[2]
    flow.env.init()
    config = flow.function_config()
    config.default_placement_scope(flow.scope.placement(args.backend, "0:0"))

    @flow.global_function("predict", function_config=config)
    def PredictNet(
        image: tp.Numpy.Placeholder((1, height, width, 3), dtype=flow.float32)
    ) -> tp.Numpy:
        style_out = style_model.styleNet(image, trainable=True, backend=args.backend)
        return style_out

    print("===============================>load begin")
    # flow.load_variables(flow.checkpoint.get(args.model_load_dir))
    flow.train.CheckPoint().load(args.model_load_dir)
    print("===============================>load end")

    import datetime

    a = datetime.datetime.now()

    print("predict begin")
    style_out = PredictNet(input_image)
    style_out = np.clip(style_out, 0, 255)
    print("predict end")

    b = datetime.datetime.now()
    c = b - a

    print("time: %s ms, height: %d, width: %d" % (c.microseconds / 1000, height, width))

    cv2.imwrite(args.output_image_path, recover_image(style_out))
    # flow.checkpoint.save("./stylenet")


def get_parser(parser=None):
    parser = argparse.ArgumentParser("flags for neural style")
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        help="gpu or cambricon"
    )
    parser.add_argument(
        "--input_image_path", type=str, default="test_img/tiger.jpg", help="image path"
    )
    parser.add_argument(
        "--output_image_path", type=str, default="test_img/tiger.jpg", help="image path"
    )
    parser.add_argument(
        "--model_load_dir", type=str, default="", help="model save directory"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
```

101. oneflow/python/test/models/styletransform/requirements.txt 
```.txt
numpy==1.17.4
opencv-python==4.2.0.32
```

102. oneflow/python/test/models/styletransform/save_style_model.py 
```.py
import os
import shutil
import argparse
import oneflow as flow
import style_model


def _init_oneflow_env_and_config():
    flow.env.init()
    flow.enable_eager_execution(False)
    flow.config.enable_legacy_model_io(True)


def _make_style_transform_predict_func(width, height, backend="gpu"):
    batch_size = 1
    channels = 3

    func_cfg = flow.function_config()
    func_cfg.default_placement_scope(flow.scope.placement(backend, "0:0"))

    @flow.global_function("predict", function_config=func_cfg)
    def predict_fn(
        image: flow.typing.Numpy.Placeholder(
            shape=(batch_size, height, width, channels), dtype=flow.float32
        )
    ) -> flow.typing.Numpy:
        style_out = style_model.styleNet(image, backend=backend)
        return style_out

    return predict_fn


def main(args):
    _init_oneflow_env_and_config()

    predict_fn = _make_style_transform_predict_func(args.image_width, args.image_height, args.backend)
    flow.train.CheckPoint().load(args.model_dir)
    # flow.load_variables(flow.checkpoint.get(args.model_dir))
    print("predict_fn construct finished")

    saved_model_path = args.save_dir
    model_version = args.model_version

    model_version_path = os.path.join(saved_model_path, str(model_version))
    if os.path.exists(model_version_path) and os.path.isdir(model_version_path):
        if args.force_save:
            print(
                f"WARNING: The model version path '{model_version_path}' already exist"
                ", old version directory will be replaced"
            )
            shutil.rmtree(model_version_path)
        else:
            raise ValueError(
                f"The model version path '{model_version_path}' already exist"
            )

    saved_model_builder = (
        flow.saved_model.ModelBuilder(saved_model_path)
        .ModelName(args.model_name)
        .Version(model_version)
    )
    saved_model_builder.AddFunction(predict_fn).Finish()
    saved_model_builder.Save()


def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        help="gpu or cambricon"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="stylenet_nhwc",
        help="model parameters directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="style_transform_models",
        help="directory to save models",
    )
    parser.add_argument(
        "--model_name", type=str, default="style_transform", help="model name"
    )
    parser.add_argument("--model_version", type=int, default=1, help="model version")
    parser.add_argument(
        "--force_save",
        default=False,
        action="store_true",
        help="force save model whether already exists or not",
    )
    parser.add_argument(
        "--image_width", type=int, default=640, help="input image width"
    )
    parser.add_argument(
        "--image_height", type=int, default=640, help="input image height"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
```

103. oneflow/python/test/models/styletransform/save_style_model.sh
```.sh
#!/bin/bash
set -ex

# download model parameters for first-time 
# wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/styletranform.tar.gz
# tar zxvf styletranform.tar.gz

base_dir=`dirname $0`
BACKEND="${1:-gpu}"
SERVING_MODEL_NAME="style_transform_models_${BACKEND}"

python3 $base_dir/save_style_model.py \
    --backend $BACKEND \
    --model_dir stylenet_nhwc \
    --save_dir $SERVING_MODEL_NAME \
    --model_version 1 \
    --image_width 640 \
    --image_height 640 \
    --force_save
```

104. oneflow/python/test/models/styletransform/style_model.py
```.py
import oneflow as flow


def instance_norm_cambricon(input, name_prefix, trainable=True):
    out, mean, var = flow.nn.InstanceNorm2d(input, name=name_prefix)
    return out

def instance_norm_gpu(input, name_prefix, trainable = True):
    (mean, variance) = flow.nn.moments(input, [1, 2], keepdims = True)
    gamma = flow.get_variable(
        name_prefix + "_gamma",
        shape = (1, 1, 1, input.shape[3]),
        dtype=input.dtype,
        initializer = flow.ones_initializer(),
        trainable = trainable
    )
    beta = flow.get_variable(
        name_prefix + "_beta",
        shape = (1, 1, 1, input.shape[3]),
        dtype=input.dtype,
        initializer = flow.zeros_initializer(),
        trainable = trainable
    )
    epsilon = 1e-5
    normalized = (input - mean) / flow.math.sqrt(variance + epsilon)
    return gamma * normalized + beta

def instance_norm(input, name_prefix, trainable=True, backend="gpu"):
    if backend == "gpu":
        return instance_norm_gpu(input, name_prefix, trainable)
    elif backend == "cambricon":
        return instance_norm_cambricon(input, name_prefix, trainable)
    else:
        return None

def conv2d_layer(
    name,
    input,
    out_channel,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NHWC",
    dilation_rate=1,
    use_bias=True,
    weight_initializer=flow.variance_scaling_initializer(
        2, "fan_out", "random_normal", data_format="NHWC"
    ),
    bias_initializer=flow.zeros_initializer(),
    trainable=True,
):
    weight_shape = (out_channel, kernel_size, kernel_size, input.shape[3])
    weight = flow.get_variable(
        name + "_weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        trainable=trainable,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "_bias",
            shape=(out_channel,),
            dtype=input.dtype,
            initializer=bias_initializer,
            trainable=trainable,
        )
        output = flow.nn.bias_add(output, bias, data_format)
    return output


def upsampleConvLayer(
    input,
    name_prefix,
    channel,
    kernel_size,
    hw_scale=(2, 2),
    data_format="NHWC",
    # interpolation = "bilinear",
    interpolation="nearest",
    trainable=True,
):
    upsample = flow.layers.upsample_2d(
        input,
        size=hw_scale,
        data_format=data_format,
        interpolation=interpolation,
        name=name_prefix + "_%s" % interpolation,
    )
    return conv2d_layer(
        name_prefix + "_conv",
        upsample,
        channel,
        kernel_size=kernel_size,
        strides=1,
        trainable=trainable,
    )


def resBlock(input, channel, name_prefix, trainable=True, backend="gpu"):
    out = conv2d_layer(
        name_prefix + "_conv1",
        input,
        channel,
        kernel_size=3,
        strides=1,
        trainable=trainable,
    )
    out = instance_norm(out, name_prefix + "_in1", trainable=trainable, backend=backend)
    out = flow.nn.relu(out)
    out = conv2d_layer(
        name_prefix + "_conv2",
        out,
        channel,
        kernel_size=3,
        strides=1,
        trainable=trainable,
    )
    out = instance_norm(out, name_prefix + "_in2", trainable=trainable, backend=backend)
    return out + input


def styleNet(input, trainable=True, backend="gpu"):
    with flow.scope.namespace("style_transfer"):
        # Initial convolution layers
        conv1 = conv2d_layer(
            "first_conv", input, 32, kernel_size=9, strides=1, trainable=trainable
        )
        in1 = instance_norm(conv1, "first_conv_in", backend=backend)
        in1 = flow.nn.relu(in1)
        conv2 = conv2d_layer(
            "second_conv", in1, 64, kernel_size=3, strides=2, trainable=trainable
        )
        in2 = instance_norm(conv2, "second_conv_in", backend=backend)
        in2 = flow.nn.relu(in2)
        conv3 = conv2d_layer(
            "third_conv", in2, 128, kernel_size=3, strides=2, trainable=trainable
        )
        in3 = instance_norm(conv3, "third_conv_in", trainable=trainable, backend=backend)
        in3 = flow.nn.relu(in3)
        # Residual layers
        res1 = resBlock(in3, 128, "res1", trainable=trainable, backend=backend)
        res2 = resBlock(res1, 128, "res2", trainable=trainable, backend=backend)
        res3 = resBlock(res2, 128, "res3", trainable=trainable, backend=backend)
        res4 = resBlock(res3, 128, "res4", trainable=trainable, backend=backend)
        res5 = resBlock(res4, 128, "res5", trainable=trainable, backend=backend)
        # Upsampling Layers
        upsample1 = upsampleConvLayer(res5, "upsample1", 64, 3, trainable=trainable)
        # upsample1 = deconv(res5, 64, "upsample1", kernel_size = 4, strides = [2, 2], trainable = True)
        in4 = instance_norm(upsample1, "upsample1_in", trainable=trainable, backend=backend)
        in4 = flow.nn.relu(in4)
        upsample2 = upsampleConvLayer(in4, "upsample2", 32, 3, trainable=trainable)
        # upsample2 = deconv(in4, 32, "upsample2", kernel_size = 4, strides = [2, 2], trainable = True)
        in5 = instance_norm(upsample2, "upsample2_in", trainable=trainable, backend=backend)
        in5 = flow.nn.relu(in5)
        out = conv2d_layer(
            "last_conv", in5, 3, kernel_size=9, strides=1, trainable=trainable
        )
        # out = flow.clamp(conv1, 0, 255)
        # print('out.shape', out.shape)
    return out


def mse_loss(input):
    return flow.math.reduce_mean(flow.math.square(input))
```



## [MLU270 算子库移植代码](https://github.com/wanghongsheng01/framework_cambricaon/tree/master/oneflow_cambricon-cambricon/oneflow/python/test/ops/)

105. oneflow/python/test/ops/_test_add_n_cambricon.py

106. oneflow/python/test/ops/_test_bias_add_cambricon.py

107. oneflow/python/test/ops/_test_bn_cambricon.py

108.  oneflow/python/test/ops/_test_cambricon_upsample.py

109. oneflow/python/test/ops/_test_concat_cambricon.py

110. oneflow/python/test/ops/_test_conv_cambricon.py

111. oneflow/python/test/ops/_test_instance_norm_2d_cambricon.py

112. oneflow/python/test/ops/_test_matmul_cambricon.py

113. oneflow/python/test/ops/_test_pool_cambricon.py 

114. oneflow/python/test/ops/_test_prelu_cambricon.py

115. oneflow/python/test/ops/_test_relu_cambricon.py

116. oneflow/python/test/ops/_test_sigmoid_cambricon.py

117. oneflow/python/test/ops/_test_softmax_cambricon.py

118. oneflow/python/test/ops/_test_transpose_cambricon.py 

119. oneflow/python/test/ops/test_activations.py

120. oneflow/python/test/ops/test_argmax.py

121. oneflow/python/test/ops/test_argsort.py 

122. oneflow/python/test/ops/test_batch_normalization.py

123. oneflow/python/test/ops/test_bias_add.py

124. oneflow/python/test/ops/test_broadcast_like.py

125. oneflow/python/test/ops/test_broadcast_normal.py

126. oneflow/python/test/ops/test_cast.py 

127. oneflow/python/test/ops/test_combined_margin_loss.py

128. oneflow/python/test/ops/test_compat_conv2d.py 

129. oneflow/python/test/ops/test_concat.py

130. oneflow/python/test/ops/test_constant.py

131. oneflow/python/test/ops/test_ctc_loss.py

132. oneflow/python/test/ops/test_dropout.py

133. oneflow/python/test/ops/test_expand_dims.py

134. oneflow/python/test/ops/test_flatten.py

135. oneflow/python/test/ops/test_fuse_cast_scale.py

136. oneflow/python/test/ops/test_in_top_k.py 

137. oneflow/python/test/ops/test_l2_normalize.py

138.  oneflow/python/test/ops/test_lamb.py 

139. oneflow/python/test/ops/test_layers_conv1d.py 

140. oneflow/python/test/ops/test_layers_conv2d.py

141. oneflow/python/test/ops/test_layers_conv3d.py

142.  oneflow/python/test/ops/test_leaky_relu.py

143.  oneflow/python/test/ops/test_matmul.py

144.  oneflow/python/test/ops/test_memory_zone_out_of_memory.py 

145.  oneflow/python/test/ops/test_moments.py 

146.  oneflow/python/test/ops/test_multi_optimizer.py

147.  oneflow/python/test/ops/test_nn_conv1d.py

148.  oneflow/python/test/ops/test_nn_conv2d.py

149.  oneflow/python/test/ops/test_nn_conv2d_padding.py 

150. oneflow/python/test/ops/test_nn_conv2d_padding_dynamic.py

151.  oneflow/python/test/ops/test_nn_conv3d.py 
152.  oneflow/python/test/ops/test_optimizers.py
153.  oneflow/python/test/ops/test_partial_fc.py 
154.  oneflow/python/test/ops/test_prelu.py
155.  oneflow/python/test/ops/test_quantize_op.py
156.  oneflow/python/test/ops/test_random_mask_like.py 
157.  oneflow/python/test/ops/test_reduce_mean.py 
158.  oneflow/python/test/ops/test_reduce_ops.py
159.  oneflow/python/test/ops/test_reduce_opsV2.py 
160.  oneflow/python/test/ops/test_reduce_sum.py
161.  oneflow/python/test/ops/test_relu_fakedev.py
162.  oneflow/python/test/ops/test_reshapeV2.py
163.  oneflow/python/test/ops/test_reshapeV3.py
164.  oneflow/python/test/ops/test_scalar_by_tensor_ops.py
165.  oneflow/python/test/ops/test_shuffle.py
166.  oneflow/python/test/ops/test_sigmoid_cross_entropy.py
167.  oneflow/python/test/ops/test_smooth_l1_loss.py 
168.  oneflow/python/test/ops/test_softmax.py
169.  oneflow/python/test/ops/test_softmax_cross_entropy.py
170.  oneflow/python/test/ops/test_sort.py
171.  oneflow/python/test/ops/test_sparse_cross_entropy.py
172.  oneflow/python/test/ops/test_sparse_cross_entropy_ms.py
173.  oneflow/python/test/ops/test_sparse_softmax_cross_entropy.py
174.  oneflow/python/test/ops/test_sparse_softmax_cross_entropy_ms.py
175.  oneflow/python/test/ops/test_split_like.py
176.  oneflow/python/test/ops/test_square.py
177.  oneflow/python/test/ops/test_squeeze.py
178.  oneflow/python/test/ops/test_top_k.py
179.  oneflow/python/test/ops/test_transpose.py
180.  oneflow/python/test/ops/test_upsample.py
181.  oneflow/python/test/ops/test_util.py
182.  oneflow/python/test/ops/test_watch_diff.py



## [MLU270 算子库 kernel 移植代码](https://github.com/wanghongsheng01/framework_cambricaon/tree/master/oneflow_cambricon-cambricon/oneflow/user/kernels)

183.  oneflow/user/kernels/add_n_kernel_mlu.cpp
184.  oneflow/user/kernels/bias_add_mlu_kernel.cpp
185.  oneflow/user/kernels/concat_kernel_mlu.cpp
186.  oneflow/user/kernels/conv_kernel_cambricon.cpp
187.  oneflow/user/kernels/instance_norm_kernel_cambricon.cpp
188.  oneflow/user/kernels/normalization_kernel_mlu.cpp 
189.  oneflow/user/kernels/pool_cambricon_kernel.cpp
190.  oneflow/user/kernels/prelu_kernel_mlu.cpp 
191.  oneflow/user/kernels/relu_kernel_fakedev.cpp
192.  oneflow/user/kernels/relu_kernel_mlu.cpp
193.  oneflow/user/kernels/reshape_kernel.cpp 
194.  oneflow/user/kernels/sigmoid_kernel_mlu.cpp
195.  oneflow/user/kernels/transpose_kernel_mlu.cpp
196.  oneflow/user/kernels/upsample_kernel_mlu.cpp
197.  oneflow/user/ops/instance_norm_op.cpp
198.  oneflow/user/ops/upsample_op.cpp



