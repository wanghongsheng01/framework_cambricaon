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
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

namespace {

// MemCaseId int64_t encode
// |              | device_type | device_index  |                         |
// |              | ---- 5 ---- | ----- 7 ----- |                         |
// |              |         MemZoneId           |   pglck   | reg_by_net  |
// |              | ----------- 12 ------------ | --- 5 --- | ---- 1 ---- |
// |   reserved   |                       MemCaseId                       |
// | ---- 46 ---- | ------------------------ 18 ------------------------- |
// | ----------------------------- 64 bit ------------------------------- |

// GlobalMemCaseId int64_t encode
// |          |   rank   | MemCaseId  |
// |          | -- 19 -- | --- 18 --- |
// | reserved |    GlobalMemCaseId    |
// | -- 27 -- | -------- 37 --------- |
// | ------------ 64 bit ------------ |

constexpr size_t kRegByNetBits = 1;
constexpr size_t kPageLockedTypeBits = 5;
constexpr size_t kDeviceIndexBits = MemZoneId::kDeviceIndexBits;
constexpr size_t kDeviceTypeBits = MemZoneId::kDeviceTypeBits;

constexpr size_t kPageLockedTypeShift = kRegByNetBits;
constexpr size_t kDeviceIndexShift = kPageLockedTypeShift + kPageLockedTypeBits;
constexpr size_t kDeviceTypeShift = kDeviceIndexShift + kDeviceIndexBits;
constexpr size_t kRankShift = kDeviceTypeShift + kDeviceTypeBits;

}  // namespace

MemCaseId::MemCaseId(const MemoryCase& mem_case) {
  // TODO: consider migrate to registry
  DeviceType device_type = DeviceType::kInvalidDevice;
  device_index_t device_index = 0;
  DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
  bool host_mem_registered_by_network = false;
  if (mem_case.has_host_mem()) {
    device_type = DeviceType::kCPU;
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      page_locked_device_type = DeviceType::kGPU;
      device_index = mem_case.host_mem().cuda_pinned_mem().device_id();
    } else if (mem_case.host_mem().has_fake_dev_pinned_mem()) {
      page_locked_device_type = DeviceType::kFAKEDEVICE;
    } else {
      // host mem is pageable
    }
    if (mem_case.host_mem().has_used_by_network() && mem_case.host_mem().used_by_network()) {
      host_mem_registered_by_network = true;
    }
  } else if (mem_case.has_device_cuda_mem()) {
    device_type = DeviceType::kGPU;
    device_index = mem_case.device_cuda_mem().device_id();
  } else if (mem_case.has_fake_dev_mem()) {
    device_type = DeviceType::kFAKEDEVICE;
  } else if (mem_case.has_device_cambricon_mem()) {
    device_type = DeviceType::kCambricon;
    device_index = mem_case.device_cambricon_mem().device_id();
  } else {
    // Uninitialized MemoryCase, all member are set to default
  }
  mem_zone_id_ = MemZoneId{device_type, device_index};
  host_mem_page_locked_device_type_ = page_locked_device_type;
  host_mem_registered_by_network_ = host_mem_registered_by_network;
}

int64_t SerializeMemCaseIdToInt64(const MemCaseId& mem_case_id) {
  int64_t id = static_cast<int64_t>(mem_case_id.is_host_mem_registered_by_network());
  id |= static_cast<int64_t>(mem_case_id.host_mem_page_locked_device_type())
        << kPageLockedTypeShift;
  id |= static_cast<int64_t>(mem_case_id.mem_zone_id().device_index()) << kDeviceIndexShift;
  id |= static_cast<int64_t>(mem_case_id.mem_zone_id().device_type()) << kDeviceTypeShift;
  return id;
}

int64_t SerializeGlobalMemCaseIdToInt64(const GlobalMemCaseId& global_mem_case_id) {
  int64_t id = SerializeMemCaseIdToInt64(global_mem_case_id.mem_case_id());
  id |= static_cast<int64_t>(global_mem_case_id.rank()) << kRankShift;
  return id;
}

void SerializeMemCaseIdToMemCase(const MemCaseId& mem_case_id, MemoryCase* mem_case) {
  // TODO: consider migrate to registry
  if (mem_case_id.mem_zone_id().device_type() == DeviceType::kCPU) {
    auto* host_mem = mem_case->mutable_host_mem();
    if (mem_case_id.host_mem_page_locked_device_type() == DeviceType::kGPU) {
      host_mem->mutable_cuda_pinned_mem()->set_device_id(mem_case_id.mem_zone_id().device_index());
    } else if (mem_case_id.host_mem_page_locked_device_type() == DeviceType::kFAKEDEVICE) {
      host_mem->mutable_fake_dev_pinned_mem();
    } else {
      host_mem->Clear();
    }
    if (mem_case_id.is_host_mem_registered_by_network()) { host_mem->set_used_by_network(true); }
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(mem_case_id.mem_zone_id().device_index());
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kFAKEDEVICE) {
    mem_case->mutable_fake_dev_mem();
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kCambricon) {
    mem_case->mutable_device_cambricon_mem()->set_device_id(
        mem_case_id.mem_zone_id().device_index());
  } else {
    UNIMPLEMENTED();
  }
}

// Patch the source memory case to destination memory case.
// Patch failed when src_mem_case and dst_mem_case have different device_type
// or one of them has invalid device_type.
// Patch failed when src_mem_case and dst_mem_case have the same non-cpu device_type
// but have different device_index.
// When src_mem_case and dst_mem_case have the same cpu device_type
// and src_mem_case has more constrain than dst_mem_case(page-locked by other device,
// such as gpu or network device), patch the constrain of src_mem_case to dst_mem_case.
bool PatchMemCaseId(MemCaseId* dst_mem_case_id, const MemCaseId& src_mem_case_id) {
  DeviceType device_type = src_mem_case_id.mem_zone_id().device_type();
  if (device_type == DeviceType::kInvalidDevice) { return false; }
  if (device_type != dst_mem_case_id->mem_zone_id().device_type()) { return false; }

  if (device_type == DeviceType::kCPU) {
    MemCaseId::device_index_t device_index = dst_mem_case_id->mem_zone_id().device_index();
    auto page_locked_device_type = dst_mem_case_id->host_mem_page_locked_device_type();
    bool registered_by_network = dst_mem_case_id->is_host_mem_registered_by_network();
    if (src_mem_case_id.host_mem_page_locked_device_type() == DeviceType::kGPU) {
      page_locked_device_type = DeviceType::kGPU;
      device_index = src_mem_case_id.mem_zone_id().device_index();
    } else if (src_mem_case_id.host_mem_page_locked_device_type() == DeviceType::kFAKEDEVICE) {
      page_locked_device_type = DeviceType::kFAKEDEVICE;
    } else {
      // do nothing
    }
    if (src_mem_case_id.is_host_mem_registered_by_network()) { registered_by_network = true; }
    *dst_mem_case_id =
        MemCaseId{device_type, device_index, page_locked_device_type, registered_by_network};
  } else {
    if (dst_mem_case_id->mem_zone_id().device_index()
        != src_mem_case_id.mem_zone_id().device_index()) {
      return false;
    }
  }
  return true;
}

bool PatchMemCase(MemoryCase* dst_mem_case, const MemoryCase& src_mem_case) {
  MemCaseId src_mem_case_id{src_mem_case};
  MemCaseId dst_mem_case_id{*dst_mem_case};
  bool result = PatchMemCaseId(&dst_mem_case_id, src_mem_case_id);
  SerializeMemCaseIdToMemCase(dst_mem_case_id, dst_mem_case);
  return result;
}

MemCaseId GenerateCorrespondingPageLockedHostMemCaseId(const MemCaseId& mem_case_id) {
  CHECK_NE(mem_case_id.mem_zone_id().device_type(), DeviceType::kInvalidDevice);
  CHECK_NE(mem_case_id.mem_zone_id().device_type(), DeviceType::kCPU);
  DeviceType page_locked_device_type = DeviceType::kInvalidDevice;
  MemCaseId::device_index_t device_index = 0;
  if (mem_case_id.mem_zone_id().device_type() == DeviceType::kGPU) {
    page_locked_device_type = DeviceType::kGPU;
    device_index = mem_case_id.mem_zone_id().device_index();
  } else if (mem_case_id.mem_zone_id().device_type() == DeviceType::kFAKEDEVICE) {
    page_locked_device_type = DeviceType::kFAKEDEVICE;
  } else {
    // do nothing
  }
  return MemCaseId{DeviceType::kCPU, device_index, page_locked_device_type};
}

MemoryCase GenerateCorrespondingPageLockedHostMemoryCase(const MemoryCase& mem_case) {
  MemCaseId host_mem_case_id = GenerateCorrespondingPageLockedHostMemCaseId(MemCaseId{mem_case});
  MemoryCase host_mem_case;
  SerializeMemCaseIdToMemCase(host_mem_case_id, &host_mem_case);
  return host_mem_case;
}

}  // namespace oneflow
