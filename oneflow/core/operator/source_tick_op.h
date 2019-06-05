#ifndef ONEFLOW_CORE_OPERATOR_SOURCE_TICK_OP_H_
#define ONEFLOW_CORE_OPERATOR_SOURCE_TICK_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class SourceTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SourceTickOp);
  SourceTickOp() = default;
  ~SourceTickOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  LogicalNode* NewProperLogicalNode() const override { return new SourceTickLogicalNode; }

 private:
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOURCE_TICK_OP_H_
