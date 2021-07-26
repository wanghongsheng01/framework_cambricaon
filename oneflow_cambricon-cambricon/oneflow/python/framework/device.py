"""
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
"""
import oneflow.core.common.device_type_pb2 as device_type_pb
from oneflow.python.oneflow_export import oneflow_export


_device_tag_2_device_type = {
    "cpu": device_type_pb.kCPU,
    "gpu": device_type_pb.kGPU,
    "dummy": device_type_pb.kFAKEDEVICE,
    "cambricon": device_type_pb.kCambricon,
}


_device_type_2_device_tag = {
    device_type_pb.kCPU: "cpu",
    device_type_pb.kGPU: "gpu",
    device_type_pb.kFAKEDEVICE: "dummy",
    device_type_pb.kCambricon: "cambricon",
}


@oneflow_export("is_valid_device_tag")
def is_valid_device_tag(device_tag: str):
    if not isinstance(device_tag, str):
        return False
    return device_tag in _device_tag_2_device_type
