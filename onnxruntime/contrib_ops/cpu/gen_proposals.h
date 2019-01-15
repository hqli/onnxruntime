// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {
template <typename T>
class GenerateProposals final : public OpKernel {
 public:
  explicit GenerateProposals(const OpKernelInfo& info) : OpKernel(info) {
    // min size
    float min_size_tmp;
    if (info.GetAttr<float>("min_size", &min_size_tmp).IsOK()) {
      min_size_ = min_size_tmp;
    }

    // nms_thresh
    float nms_thresh_tmp;
    if (info.GetAttr<float>("nms_thresh", &nms_thresh_tmp).IsOK()) {
      nms_thresh_ = nms_thresh_tmp;
    }

    // post_nms_topN
    int64_t post_nms_topN_tmp;
    if (info.GetAttr<int64_t>("post_nms_topN", &post_nms_topN_tmp).IsOK()) {
      post_nms_topN_ = post_nms_topN_tmp;
    }

    // pre_nms_topN
    int64_t pre_nms_topN_tmp;
    if (info.GetAttr<int64_t>("pre_nms_topN", &pre_nms_topN_tmp).IsOK()) {
      pre_nms_topN_ = pre_nms_topN_tmp;
    }

    // spatial_scale
    float spatial_scale_tmp;
    if (info.GetAttr<float>("spatial_scale", &spatial_scale_tmp).IsOK()) {
      spatial_scale_ = spatial_scale_tmp;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float min_size_;  // TODO why is this float?
  float nms_thresh_;
  int64_t post_nms_topN_;
  int64_t pre_nms_topN_;
  float spatial_scale_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GenerateProposals);
};
}  // namespace contrib
}  // namespace onnxruntime
