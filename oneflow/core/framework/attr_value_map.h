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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"

namespace oneflow {

using AttrName2AttrVal = HashMap<std::string, std::shared_ptr<const AttrVal>>;

class AttrValueMap {
 public:
  explicit AttrValueMap(const std::shared_ptr<const AttrName2AttrVal>& map): map_(map) {}
  AttrValueMap(const AttrValueMap&) = default;
  AttrValueMap(AttrValueMap&&) = default;
  ~AttrValueMap() = default;

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const AttrName2AttrVal& map() const { return *map_; }

 private:
  std::shared_ptr<const AttrName2AttrVal> map_;
};

class ComposedAttrValueMap final {
 public:
  ComposedAttrValueMap(const AttrValueMap& base): base_(base) {}
  ComposedAttrValueMap(const ComposedAttrValueMap&) = delete;
  ComposedAttrValueMap(ComposedAttrValueMap&&) = delete;
  ~ComposedAttrValueMap() = default;

  void reset_prior(const AttrValueMap& prior) { prior_ = prior; }

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

 private:
  const AttrValueMap base_;
  AttrValueMap prior_;
};

class MutableAttrValueMap : public HashMap<std::string, std::shared_ptr<cfg::AttrValue>> {
 public:
  MutableAttrValueMap(); 

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);

  const std::shared_ptr<AttrName2AttrVal>& FreeseAndGetMap();

 priavate:
  std::shared_ptr<AttrName2AttrVal> map_; 
  bool frozen_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_
