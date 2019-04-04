#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

namespace {

void CheckBlobInRegstNotDisabled(const RegstDescProto& regst_desc) {
  CHECK(regst_desc.regst_desc_type().has_data_regst_desc());
  CHECK(regst_desc.regst_desc_type().data_regst_desc().packed_blob_desc().is_body_disabled() == false);
}

} // namespace

RegstMgr::RegstMgr(const Plan& plan) {
  std::list<const RegstDescProto*> regst_protos;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    for (const auto& pair : task.produced_regst_desc()) { regst_protos.push_back(&pair.second); }
  }
  InitFromRegstProtoList(regst_protos);
}

RegstMgr::RegstMgr(const std::list<const RegstDescProto*>& regst_protos) {
  InitFromRegstProtoList(regst_protos);
}

void RegstMgr::InitFromRegstProtoList(const std::list<const RegstDescProto*>& regst_protos) {
  std::vector<const RegstDescProto*> sorted_regst_protos(regst_protos.begin(), regst_protos.end());
  for (const RegstDescProto* regst_desc : regst_protos) {
    CHECK(
        regst_desc_id2rt_regst_desc_
            .emplace(regst_desc->regst_desc_id(), std::make_unique<const RtRegstDesc>(*regst_desc))
            .second);
    if (regst_desc->mem_shared_id() != -1) { CHECK_EQ(regst_desc->register_num(), 1); }
  }
  auto GetMemSize4Regst = [&](const RegstDescProto* regst_desc) {
    if (regst_desc->mem_shared_id() == -1) {
      return regst_desc_id2rt_regst_desc_.at(regst_desc->regst_desc_id())->TotalMainByteSize4AllRegst();
    } else {
      CheckBlobInRegstNotDisabled(*regst_desc);
      return regst_desc_id2rt_regst_desc_.at(regst_desc->regst_desc_id())
                 ->TotalMainByteSize4AllRegst()
             + regst_desc->mem_shared_offset();
    }
  };
  std::sort(sorted_regst_protos.begin(), sorted_regst_protos.end(),
            [&](const RegstDescProto* lhs, const RegstDescProto* rhs) {
              return (lhs->mem_shared_id() < rhs->mem_shared_id())
                     || (lhs->mem_shared_id() == rhs->mem_shared_id()
                         && GetMemSize4Regst(lhs) > GetMemSize4Regst(rhs));
            });
  int32_t last_mem_shared_id = -1;
  char* main_mem_ptr = nullptr;
  for (const RegstDescProto* regst_desc : sorted_regst_protos) {
    if (regst_desc->regst_desc_type().has_data_regst_desc() == false) { continue; }
    size_t mem_size = GetMemSize4Regst(regst_desc);
    int32_t current_mem_shared_id = regst_desc->mem_shared_id();
    if (current_mem_shared_id == -1 || (current_mem_shared_id != last_mem_shared_id)) {
      if (mem_size == 0) {
        main_mem_ptr = nullptr;
      } else {
        main_mem_ptr = Global<MemoryAllocator>::Get()->Allocate(regst_desc->mem_case(),
                                                                GetMemSize4Regst(regst_desc));
      }
    }
    CHECK(regst_desc_id2main_mem_ptr_.emplace(regst_desc->regst_desc_id(), main_mem_ptr).second);
    last_mem_shared_id = current_mem_shared_id;
  }
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  const RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_.at(regst_desc_id).get();
  char* main_mem_ptr = nullptr;
  if (regst_desc_id2main_mem_ptr_.find(regst_desc_id) != regst_desc_id2main_mem_ptr_.end()) {
    main_mem_ptr = regst_desc_id2main_mem_ptr_.at(regst_desc_id);
  }
  std::vector<LbiBlobDescPair> lbi_pairs;
  if (regst_desc_type.has_data_regst_desc()) {
    for (const LbiBlobDescPair& pair : regst_desc_type.data_regst_desc().lbi2blob_desc()) {
      lbi_pairs.push_back(pair);
    }
    std::sort(lbi_pairs.begin(), lbi_pairs.end(), &CompareLbiBlobDescPair);
    CHECK(!lbi_pairs.empty());
  }
  if (regst_desc_proto.mem_shared_id() != -1) {
    CheckBlobInRegstNotDisabled(regst_desc_proto);
    main_mem_ptr += regst_desc_proto.mem_shared_offset();
  }
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst;
    regst->set_regst_desc(rt_regst_desc);
    if (regst_desc_type.has_data_regst_desc()) {
      NewBlobsInOneRegst(lbi_pairs, regst, rt_regst_desc, main_mem_ptr);
      if (rt_regst_desc->mem_case().has_host_mem()
          && rt_regst_desc->mem_case().host_mem().used_by_network()) {
        CheckBlobInRegstNotDisabled(regst_desc_proto);
        regst->comm_net_token_ = Global<CommNet>::Get()->RegisterMemory(
            main_mem_ptr, rt_regst_desc->MainByteSize4OneRegst());
      }
      if (main_mem_ptr != nullptr) {
        main_mem_ptr += rt_regst_desc->MainByteSize4OneRegst();
      }
    } else if (regst_desc_type.has_ctrl_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

void RegstMgr::NewBlobsInOneRegst(const std::vector<LbiBlobDescPair>& lbis, Regst* regst,
                                  const RtRegstDesc* rt_regst_desc, char* main_mem_ptr) {
  size_t separated_mem_size = rt_regst_desc->SeparatedByteSize4OneRegst();
  const RtBlobDesc* packed_blob_desc = rt_regst_desc->packed_blob_desc();
  char* cur_body_pointer = nullptr;
  char* cur_header_pointer = nullptr;
  if (separated_mem_size > 0) {
    MemoryCase host_mem_case;
    host_mem_case.mutable_host_mem();
    char* separated_mem_ptr =
        Global<MemoryAllocator>::Get()->Allocate(host_mem_case, separated_mem_size);
    regst->packed_blob_.reset(new Blob(regst, packed_blob_desc, separated_mem_ptr, main_mem_ptr));
    cur_header_pointer = separated_mem_ptr;
    cur_body_pointer = main_mem_ptr;
  } else {
    regst->packed_blob_.reset(new Blob(regst, packed_blob_desc, main_mem_ptr));
    cur_header_pointer = main_mem_ptr;
    if (main_mem_ptr == nullptr) {
      cur_body_pointer = nullptr;
    } else {
      cur_body_pointer = main_mem_ptr + packed_blob_desc->ByteSizeOfBlobHeader();
    }
  }
  rt_regst_desc->ForEachBlobDescOffsetInOnRegst(
      lbis, [&](const LbiBlobDescPair& lbi, int64_t body_offset, int64_t header_offset) {
        const RtBlobDesc* blob_desc = rt_regst_desc->GetRtBlobDescFromLbi(lbi.lbi());
        std::unique_ptr<Blob> blob_ptr;
        if (cur_body_pointer == nullptr) {
          CHECK(rt_regst_desc->is_body_disabled());
          blob_ptr = std::move(std::make_unique<Blob>(
              regst, blob_desc, cur_header_pointer + header_offset, nullptr));
        } else {
          CHECK(rt_regst_desc->is_body_disabled() == false);
          blob_ptr = std::move(std::make_unique<Blob>(
              regst, blob_desc, cur_header_pointer + header_offset, cur_body_pointer + body_offset));
          InitOFRecordBlobIfNeed(blob_ptr.get());
        }
        CHECK(regst->lbi2blob_.emplace(lbi.lbi(), std::move(blob_ptr)).second);
      });
}

void RegstMgr::InitOFRecordBlobIfNeed(Blob* blob_ptr) {
  const RtBlobDesc& blob_desc = blob_ptr->blob_desc();
  if (blob_desc.data_type() == kOFRecord) {
    int64_t elem_cnt = blob_desc.shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      Global<MemoryAllocator>::Get()->PlacementNew(&blob_ptr->mut_dptr<OFRecord>()[idx]);
    }
  }
}

const RtRegstDesc& RegstMgr::RegstDesc4RegstDescId(int64_t regst_desc_id) const {
  const auto& it = regst_desc_id2rt_regst_desc_.find(regst_desc_id);
  CHECK(it != regst_desc_id2rt_regst_desc_.end());
  return *it->second;
}

}  // namespace oneflow
