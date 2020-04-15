#ifndef ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"

namespace oneflow {
namespace vm {

class VirtualMachine;
class InstructionMsg;

class ControlStreamType final : public StreamType {
 public:
  ControlStreamType() = default;
  ~ControlStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                int64_t this_machine_id) const override;
  ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const override;
  void Compute(InstrChain* instr_chain) const override;

  bool SharingVirtualMachineThread() const override { return true; }
  void Infer(VirtualMachine* vm, InstrChain* instr_chain) const override;
  void Compute(VirtualMachine* vm, InstrChain* instr_chain) const override;
  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override;
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
