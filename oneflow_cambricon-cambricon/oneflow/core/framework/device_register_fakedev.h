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
