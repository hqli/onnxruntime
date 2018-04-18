﻿#pragma once

#include "core/framework/op_kernel.h"

namespace Lotus {

class SpaceDepthBase : public OpKernel {
 public:
  SpaceDepthBase(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("blocksize", &blocksize_);
  }

 protected:
  int64_t blocksize_ = 1;
};

template <typename T>
class SpaceToDepth final : public SpaceDepthBase {
 public:
  SpaceToDepth(const OpKernelInfo& info) : SpaceDepthBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class DepthToSpace final : public SpaceDepthBase {
 public:
  DepthToSpace(const OpKernelInfo& info) : SpaceDepthBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  //namespace Lotus