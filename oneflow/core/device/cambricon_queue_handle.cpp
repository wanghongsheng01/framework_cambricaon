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
  cb_notifier_chan_->Send(cb_notifier); // 
}

}  // namespace oneflow
#endif  // WITH_CAMBRICON