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
