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
