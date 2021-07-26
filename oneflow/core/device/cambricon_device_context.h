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