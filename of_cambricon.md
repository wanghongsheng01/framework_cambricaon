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

55. oneflow/core/kernel/mlu_tools.h
56. oneflow/core/kernel/mlu_tools.cpp

57. oneflow/core/kernel/new_kernel_util.cpp

58.  oneflow/core/kernel/new_kernel_util_mlu.cpp

59. oneflow/core/kernel/output_kernel.cpp

60. oneflow/core/kernel/softmax_kernel_mlu.cpp

61. oneflow/core/kernel/variable_kernel.cpp 

62. oneflow/core/memory/memory_allocator.cpp

63. oneflow/core/memory/memory_case.proto

64. oneflow/core/memory/memory_case_util.h
65. oneflow/core/memory/memory_case_util.cpp

66. oneflow/core/memory/memory_fake_dev_allocator.h
67. oneflow/core/memory/memory_fake_dev_allocator.cpp

68. oneflow/core/register/register_manager.cpp

69. oneflow/core/register/runtime_register_desc.cpp

70. oneflow/core/thread/cambricon_device_thread.cpp

71. oneflow/core/thread/cambricon_device_thread.h

72. oneflow/core/thread/fake_device_thread.h
73. oneflow/core/thread/fake_device_thread.cpp

74. oneflow/core/thread/thread_context.h

75. oneflow/core/thread/thread_manager.cpp




76. oneflow/python/experimental/interface_op_read_and_write.py 

77. oneflow/python/framework/config_util.py 

78. oneflow/python/framework/device.py 

79. oneflow/python/framework/placement_context.py

80. oneflow/python/framework/placement_util.py

81. oneflow/python/framework/session_util.py

82. oneflow/python/ops/layers.py 

83. oneflow/python/ops/nn_ops.py

84. oneflow/python/serving/inference_session.py 

85. oneflow/python/serving/saved_model_builder.py

86. oneflow/python/test/models/insightface/fresnet100.py

87. oneflow/python/test/models/insightface/insightface_val.py

88. oneflow/python/test/models/insightface/save_insightface_model.py 

89. oneflow/python/test/models/insightface/save_insightface_model.sh

90. oneflow/python/test/models/insightface/validate.sh

91. oneflow/python/test/models/resnet50/README.md 

92. oneflow/python/test/models/resnet50/imagenet1000_clsidx_to_labels.py

93. oneflow/python/test/models/resnet50/infer.sh

94. oneflow/python/test/models/resnet50/infer_resnet50.py

95. oneflow/python/test/models/resnet50/resnet50_model.py

96. oneflow/python/test/models/resnet50/save_model.py

97. oneflow/python/test/models/resnet50/save_model.sh

98. oneflow/python/test/models/styletransform/README.md

99. oneflow/python/test/models/styletransform/infer.sh

100. oneflow/python/test/models/styletransform/infer_of_neural_style.py

101. oneflow/python/test/models/styletransform/requirements.txt 

102. oneflow/python/test/models/styletransform/save_style_model.py 

103. oneflow/python/test/models/styletransform/save_style_model.sh

104. oneflow/python/test/models/styletransform/style_model.py

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



