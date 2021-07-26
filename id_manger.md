# id manager 重构

## 旧的 task id 解构以及为什么需要重构

旧的 task id 是由如下 bit mask 组成：
```
  //  64 bit id design:
  //   sign | machine | thread | local_work_stream | task
  //    1   |   10    |   11   |       21          |  21
```

创建一个 task id 我们同时需要 machine id, thread id, local work stream id 这几个信息，代码如下：
```.h
int64_t IDMgr::NewTaskId(int64_t machine_id, int64_t thrd_id, int64_t local_work_stream_id) {
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  CHECK_LT(machine_thrd_id2num_of_tasks_[machine_thrd_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  CHECK_LT(local_work_stream_id, static_cast<int64_t>(1) << local_work_stream_id_bit_num_);
  return machine_thrd_id | (local_work_stream_id << task_id_bit_num_)
         | (machine_thrd_id2num_of_tasks_[machine_thrd_id]++);
}
```

其中 machine id 就是指机器的序号

thread id 是指工作线程 id，它的相关接口如下：
```.h
  int64_t GetGpuComputeThrdId(int64_t dev_phy_id) const { return dev_phy_id; }
  int64_t GetGpuH2DThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuD2HThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuNcclThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuMixThrdId(int64_t dev_phy_id) const;
  int64_t GetGpuDecodeH2DThrdId(int64_t dev_phy_id) const;
  int64_t GetCpuDeviceThrdId(int64_t dev_phy_id) const;
  int64_t CommNetThrdId() const;
  int64_t TickTockThrdId() const;
  int64_t BaseIndependentThrdId() const;
  void UpdateBaseIndependentThrdId(int64_t val);
```

它的分配方式如下：

![id_manager_deconstruction](https://user-images.githubusercontent.com/7133477/107337710-ac966000-6af5-11eb-9264-7d396e3b0e2f.png)

local work stream id 为当前线程上的工作流 id，当前线程由上述 thread id 表示，工作流 id 分为两种情况：

- TaskNode::AllocateLocalWorkStreamId()
- CopyCommNetTaskNode::AllocateLocalWorkStreamId()

在 TaskNode::AllocateLocalWorkStreamId 中直接返回0，表示 task node 的基类行为就是返回为 0 的 local work stream id。唯一重载了 AllocateLocalWorkStreamId 的是 CopyCommNetTaskNode，它是通过 `this_machine_id -> {peer_machine_id, local_work_stream_id}` 这种方式来分配 local work stream id 的，每台机器上的 commnet task node 需要为每一个对应的 peer 机器分配一个 local work stream id，并且 id 在 peer machine 上从 0 自增，可以想像有一张全局静态的链路对应 stream 的表 (如下)，通过查表，copy 的链路映射到对应的 stream 上。

![id_manager_deconstruction-commnet_local_work_stream_id](https://user-images.githubusercontent.com/7133477/107341943-95a63c80-6afa-11eb-94b9-6ce36409a2c1.png)

> 注意：IDMgr::AllocateLocalWorkStreamId 目前没看到有使用的地方，应该是作废代码

通过以上分析，我们应该能看到一些问题：

- thread id 和 stream id 抽象不精确，thread id 负责了部分 stream id 的表示工作，thread 与 stream 的关系没有捋清楚。理论上一个 stream 不能对应多个 thread (非要做需要加锁，没必要)，而一个 thread 可以对应多个 stream，目前我们只用到了一个 thread 对应一个 stream，从上面把 thread 当作 stream 在用可以看出。但 thread 的抽象不是 id manager 该关心的，它应该属于更上一层的抽象，我们得知一个 stream 的时候应该以另外的途径隐含得知了 thread 信息，不需要把 thread 体现在 id 分配上，所以新的设计直接以 stream id 代替，不再出现 thread id 的概念。
- 老的接口和 id 分配上直接强假设了只有 cpu 和 gpu 的情况，无法适应设备扩展后的情况，所以 task id 编码和相关的接口都需要修改。
- 在 task id 编码扩展出新的 field 后，老的 int64_t 作为 bit mask 显的不够用了，所以需要用 128 bit 来表示 task id 编码。

## 新的 task id 设计

新的 task id 用一个自定义结构体 TaskId 来表示，为支持多设备提供更大的 id 表示范围和更方便的访问接口，定义大概如下：

```
// TaskId encode
// | ---------------------------------------- 128 bit --------------------------------------------- |
// | -- 32 -- | --- 20 --- | ---- 12 ----- | --- 10 ---- | ---- 12 ---- | ---- 10 ---- | --- 32 --- |
// | reserved | node_index | process_index | device_type | device_index | stream_index | task_index |
// |          |          ProcessId         |                  StreamId                 |            |
// |                                           TaskId                                               |

class ProcessId {
 public:
  explicit ProcessId(int32_t val) : val_(val) {}
  operator int32_t();
  int32_t node_index() const;
  int32_t process_index() const;

 private:
  int32_t val_;
};

class StreamId {
 public:
  operator int32_t();
  DeviceType device_type();
  int32_t device_index();
  int32_t stream_index();

 private:
  int32_t val_;
};

class TaskId {
 public:
  TaskId(const ProcessId& process_id, const StreamId& stream_id, int32_t task_index);
  ProcessId process_id();
  StreamId stream_id();
  int32_t task_index();

 private:
  std::bitset<128> bits_;
};
```

proto message 中对 128 位 task_id 的支持通过提供一个新的 message struct TaskIdProto 及其相应的 operator 来支持
```
message TaskIdProto {
  required uint64 low = 1;
  required uint64 high = 2;
}
```



## 其他相关 (待补充)
