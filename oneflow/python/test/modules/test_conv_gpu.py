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
import oneflow
import oneflow.experimental as flow
import numpy as np

flow.enable_eager_execution()

# to_device = flow.device("cpu")
# m = flow.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# m.to(to_device)
# in_tensor = flow.Tensor(np.random.randn(20, 16, 50, 100), device=to_device)
# output = m(in_tensor)
# print(output)

to_device = flow.device("cuda")
in_tensor = flow.Tensor(np.random.randn(3, 3), device=to_device)
print(in_tensor)
gen = flow.Generator()
gen.manual_seed(1)
output = oneflow.F.dropout(in_tensor, 0.5, gen.gen)
print(output)
