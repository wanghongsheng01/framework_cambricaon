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

#include "oneflow/core/device/topology_descriptor.h"

namespace oneflow {

namespace device {

void TopologyDescriptor::SetCPUAffinityByPCIBusID(const std::string& bus_id) const {
  SetCPUAffinity(GetCPUAffinityByPCIBusID(bus_id));
}

void TopologyDescriptor::SetMemoryAffinityByPCIBusID(const std::string& bus_id) const {
  SetMemoryAffinity(GetMemoryAffinityByPCIBusID(bus_id));
}

}  // namespace device

}  // namespace oneflow
