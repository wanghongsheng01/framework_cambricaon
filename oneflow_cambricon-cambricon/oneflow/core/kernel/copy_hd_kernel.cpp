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
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class CopyHdKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
    Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
    out_blob->CopyValidDataContentFrom(ctx.device_ctx, in_blob);
  };
  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    BnInOp2Blob("out")->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob("in"));
  }
};

#ifdef WITH_CUDA

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCopyHdConf, DeviceType::kGPU,
                            CopyHdKernel<DeviceType::kGPU>);
#endif

#ifdef WITH_FAKE_DEVICE
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCopyHdConf, DeviceType::kFAKEDEVICE,
                            CopyHdKernel<DeviceType::kFAKEDEVICE>);
#endif

#ifdef WITH_CAMBRICON
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCopyHdConf, DeviceType::kCambricon,
                            CopyHdKernel<DeviceType::kCambricon>);
#endif

}  // namespace oneflow
