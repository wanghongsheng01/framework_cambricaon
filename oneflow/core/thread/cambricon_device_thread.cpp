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