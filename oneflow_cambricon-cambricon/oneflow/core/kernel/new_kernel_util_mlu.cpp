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
