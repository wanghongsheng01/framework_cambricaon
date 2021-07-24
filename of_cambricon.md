# oneflow_cambricon

版权所有 oneflow support cambricon <br>
[Diff between master and cambricon](https://github.com/Oneflow-Inc/oneflow_cambricon/pull/17/files)


修改的文件
1. .github/workflows/test.yml 
`github.base_ref == 'cambricon'`

2. CMakeLists.txt
`option(BUILD_CAMBRICON "" OFF)`
   
```txt
# TODO: add a cmake option
add_definitions(-DWITH_FAKE_DEVICE)
```
   
3. cmake/oneflow.cmake
```.cmake
if (BUILD_CAMBRICON)
  target_link_libraries(of_ccobj cnrt cnnl cndrv)
endif()
```

4. cmake/third_party.cmake 
```.cmake
if (BUILD_CAMBRICON)
  add_definitions(-DWITH_CAMBRICON)
  if (NOT DEFINED ENV{NEUWARE_HOME})
    message(FATAL_ERROR "Environment variable NEUWARE_HOME NOT found")
  endif()
  include_directories("$ENV{NEUWARE_HOME}/include")
  link_directories("$ENV{NEUWARE_HOME}/lib64")
endif()
```

5. docker/package/manylinux/build_wheel.py
```.py
def get_common_docker_args(
    oneflow_src_dir=None,
    cache_dir=None,
    current_dir=None,
    house_dir=None,
    use_system_proxy=True,
    cambricon=None, ## 新添加的 cambricon
):
    root = Path(cache_dir)
    child = Path(current_dir)
    assert root in child.parents
    cwd = os.getcwd()
    pwd_arg = f"-v {cwd}:{cwd}"
    cache_dir_arg = f"-v {cache_dir}:{cache_dir}"
    house_dir_arg = ""
    if house_dir:
        house_dir_arg = f"-v {house_dir}:{house_dir}"
    build_dir_arg = get_build_dir_arg(cache_dir, oneflow_src_dir)
    proxy_env_arg = get_proxy_env_args() if use_system_proxy else ""
    cambricon_arg = "--env NEUWARE_HOME=/usr/local/neuware" if cambricon else "" ## 新添加的 cambricon
    return f"-v {oneflow_src_dir}:{oneflow_src_dir} {proxy_env_arg} {pwd_arg} {cambricon_arg} {house_dir_arg} {cache_dir_arg} {build_dir_arg} -w {current_dir} --shm-size=8g"  ## 新添加的 cambricon
```
[修改细节多，详见](https://github.com/Oneflow-Inc/oneflow_cambricon/pull/17/files#diff-f421848746ad2a399bbe49e938fd01f333b9b1202a5cbca55fb5ead90d5571fd)

6. oneflow/api/python/env/env.h
```.h
#ifdef WITH_CAMBRICON
#include "cnrt.h"
#endif
```

7. oneflow/core/actor/actor.cpp
```.cpp
#include "oneflow/core/device/cambricon_device_context.h"
#include "oneflow/core/device/fake_device_device_context.h"
```

8. oneflow/core/actor/copy_hd_actor.cpp
```.cpp
#if defined(WITH_CUDA) or defined(WITH_FAKE_DEVICE) or defined(WITH_CAMBRICON)
```

9. oneflow/core/common/device_type.proto
```.proto
enum DeviceType {
  kInvalidDevice = 0;
  kCPU = 1;
  kGPU = 2;
  kFAKEDEVICE = 3;
  kCambricon = 4;
}
```

10.  oneflow/core/common/id_util.h
```.h
class MemZoneId {
 public:
  using device_index_t = uint32_t;

  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static size_t kMaxDeviceTypeVal = (size_t{1} << kDeviceTypeBits) - size_t{1};
  constexpr static device_index_t kMaxDeviceIndex =
      (device_index_t{1} << kDeviceIndexBits) - device_index_t{1};

  MemZoneId() {
    device_type_ = DeviceType::kCPU;
    device_index_ = 0;
  }
  MemZoneId(DeviceType device_type, device_index_t device_index)
      : device_type_(device_type), device_index_(device_index) {
    CHECK_LE(static_cast<size_t>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }
  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }
  bool operator==(const MemZoneId& rhs) const {
    return device_type_ == rhs.device_type_ && device_index_ == rhs.device_index_;
  }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = std::hash<size_t>{}(static_cast<size_t>(device_type_));
    HashCombine(&hash, std::hash<device_index_t>{}(device_index_));
    return hash;
  }

 private:
  DeviceType device_type_;
  device_index_t device_index_;
};
```

```.h
template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const { return mem_zone_id.hash(); }
};

```

11. oneflow/core/device/cambricon_device_context.h
12. oneflow/core/device/cambricon_device_context.cpp

13. oneflow/core/device/cambricon_device_stream_index.h
14. oneflow/core/device/cambricon_device_stream_index.cpp

15. oneflow/core/device/cambricon_queue_handle.h 
16. oneflow/core/device/cambricon_queue_handle.cpp
 
17. oneflow/core/device/device_context.h

18. oneflow/core/device/fake_device_device_context.h
19. oneflow/core/device/fake_device_device_context.cpp

20. oneflow/core/device/fake_device_stream_index.h
21. oneflow/core/device/fake_device_stream_index.cpp

22. oneflow/core/device/mlu_util.h

23. oneflow/core/framework/device_register_cambricon.h

24. oneflow/core/framework/device_register_fakedev.h 

25. oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.cpp

26. oneflow/core/graph/boxing/sub_task_graph_builder_context.h
27. oneflow/core/graph/boxing/sub_task_graph_builder_context.cpp

28. oneflow/core/graph/copy_task_node.h 
29. oneflow/core/graph/copy_task_node.cpp

30. oneflow/core/graph/exec_graph.cpp

31. oneflow/core/graph/id_serialization.h
32. oneflow/core/graph/id_serialization.cpp

33. oneflow/core/graph/logical_node.h 

34. oneflow/core/graph/slice_boxing_task_node.h
35. oneflow/core/graph/slice_boxing_task_node.cpp

36. oneflow/core/graph/task_graph.h
37. oneflow/core/graph/task_graph.cpp

38. oneflow/core/graph/task_node.h
39. oneflow/core/graph/task_node.cpp

40. oneflow/core/graph_impl/normal_forward_compute_task_node.cpp 

41. oneflow/core/job/env_global_objects_scope.cpp

42. oneflow/core/job/id_manager.h 
43. oneflow/core/job/id_manager.cpp

44. oneflow/core/job/improver.cpp

45. oneflow/core/job/inter_job_mem_sharing_util.cpp

46. oneflow/core/job/plan_util.cpp

47. oneflow/core/job/resource.proto

48. oneflow/core/job_rewriter/add_input_output_ops_pass.cpp

49. oneflow/core/kernel/copy_hd_kernel.cpp

50. oneflow/core/kernel/device_tick_kernel.cpp 

51. oneflow/core/kernel/input_kernel.cpp

52. oneflow/core/kernel/kernel.h

53. oneflow/core/kernel/kernel_util.cpp 

54. oneflow/core/kernel/matmul_kernel_mlu.cpp

55. oneflow/core/kernel/mlu_tools.h
56. oneflow/core/kernel/mlu_tools.cpp

57. oneflow/core/kernel/new_kernel_util.cpp

58.  oneflow/core/kernel/new_kernel_util_mlu.cpp

59. oneflow/core/kernel/output_kernel.cpp

60. oneflow/core/kernel/softmax_kernel_mlu.cpp

61. oneflow/core/kernel/variable_kernel.cpp 

62. oneflow/core/memory/memory_allocator.cpp

63. oneflow/core/memory/memory_case.proto

64. oneflow/core/memory/memory_case_util.h
65. oneflow/core/memory/memory_case_util.cpp

66. oneflow/core/memory/memory_fake_dev_allocator.h
67. oneflow/core/memory/memory_fake_dev_allocator.cpp

68. oneflow/core/register/register_manager.cpp

69. oneflow/core/register/runtime_register_desc.cpp

70. oneflow/core/thread/cambricon_device_thread.cpp

71. oneflow/core/thread/cambricon_device_thread.h

72. oneflow/core/thread/fake_device_thread.h
73. oneflow/core/thread/fake_device_thread.cpp

74. oneflow/core/thread/thread_context.h

75. oneflow/core/thread/thread_manager.cpp




76. oneflow/python/experimental/interface_op_read_and_write.py 

77. oneflow/python/framework/config_util.py 

78. oneflow/python/framework/device.py 

79. oneflow/python/framework/placement_context.py

80. oneflow/python/framework/placement_util.py

81. oneflow/python/framework/session_util.py

82. oneflow/python/ops/layers.py 

83. oneflow/python/ops/nn_ops.py

84. oneflow/python/serving/inference_session.py 

85. oneflow/python/serving/saved_model_builder.py

86. oneflow/python/test/models/insightface/fresnet100.py

87. oneflow/python/test/models/insightface/insightface_val.py

88. oneflow/python/test/models/insightface/save_insightface_model.py 

89. oneflow/python/test/models/insightface/save_insightface_model.sh

90. oneflow/python/test/models/insightface/validate.sh

91. oneflow/python/test/models/resnet50/README.md 

92. oneflow/python/test/models/resnet50/imagenet1000_clsidx_to_labels.py

93. oneflow/python/test/models/resnet50/infer.sh

94. oneflow/python/test/models/resnet50/infer_resnet50.py

95. oneflow/python/test/models/resnet50/resnet50_model.py

96. oneflow/python/test/models/resnet50/save_model.py

97. oneflow/python/test/models/resnet50/save_model.sh

98. oneflow/python/test/models/styletransform/README.md

99. oneflow/python/test/models/styletransform/infer.sh

100. oneflow/python/test/models/styletransform/infer_of_neural_style.py

101. oneflow/python/test/models/styletransform/requirements.txt 

102. oneflow/python/test/models/styletransform/save_style_model.py 

103. oneflow/python/test/models/styletransform/save_style_model.sh

104. oneflow/python/test/models/styletransform/style_model.py

105. oneflow/python/test/ops/_test_add_n_cambricon.py

106. oneflow/python/test/ops/_test_bias_add_cambricon.py

107. oneflow/python/test/ops/_test_bn_cambricon.py

108.  oneflow/python/test/ops/_test_cambricon_upsample.py

109. oneflow/python/test/ops/_test_concat_cambricon.py

110. oneflow/python/test/ops/_test_conv_cambricon.py

111. oneflow/python/test/ops/_test_instance_norm_2d_cambricon.py

112. oneflow/python/test/ops/_test_matmul_cambricon.py

113. oneflow/python/test/ops/_test_pool_cambricon.py 

114. oneflow/python/test/ops/_test_prelu_cambricon.py

115. oneflow/python/test/ops/_test_relu_cambricon.py

116. oneflow/python/test/ops/_test_sigmoid_cambricon.py

117. oneflow/python/test/ops/_test_softmax_cambricon.py

118. oneflow/python/test/ops/_test_transpose_cambricon.py 

119. oneflow/python/test/ops/test_activations.py

120. oneflow/python/test/ops/test_argmax.py

121. oneflow/python/test/ops/test_argsort.py 

122. oneflow/python/test/ops/test_batch_normalization.py

123. oneflow/python/test/ops/test_bias_add.py

124. oneflow/python/test/ops/test_broadcast_like.py

125. oneflow/python/test/ops/test_broadcast_normal.py

126. oneflow/python/test/ops/test_cast.py 

127. oneflow/python/test/ops/test_combined_margin_loss.py

128. oneflow/python/test/ops/test_compat_conv2d.py 

129. oneflow/python/test/ops/test_concat.py

130. oneflow/python/test/ops/test_constant.py

131. oneflow/python/test/ops/test_ctc_loss.py

132. oneflow/python/test/ops/test_dropout.py

133. oneflow/python/test/ops/test_expand_dims.py

134. oneflow/python/test/ops/test_flatten.py

135. oneflow/python/test/ops/test_fuse_cast_scale.py

136. oneflow/python/test/ops/test_in_top_k.py 

137. oneflow/python/test/ops/test_l2_normalize.py

138.  oneflow/python/test/ops/test_lamb.py 

139. oneflow/python/test/ops/test_layers_conv1d.py 

140. oneflow/python/test/ops/test_layers_conv2d.py

141. oneflow/python/test/ops/test_layers_conv3d.py

142.  oneflow/python/test/ops/test_leaky_relu.py

143.  oneflow/python/test/ops/test_matmul.py

144.  oneflow/python/test/ops/test_memory_zone_out_of_memory.py 

145.  oneflow/python/test/ops/test_moments.py 

146.  oneflow/python/test/ops/test_multi_optimizer.py

147.  oneflow/python/test/ops/test_nn_conv1d.py

148.  oneflow/python/test/ops/test_nn_conv2d.py

149.  oneflow/python/test/ops/test_nn_conv2d_padding.py 

150. oneflow/python/test/ops/test_nn_conv2d_padding_dynamic.py

151.  oneflow/python/test/ops/test_nn_conv3d.py 
152.  oneflow/python/test/ops/test_optimizers.py
153.  oneflow/python/test/ops/test_partial_fc.py 
154.  oneflow/python/test/ops/test_prelu.py
155.  oneflow/python/test/ops/test_quantize_op.py
156.  oneflow/python/test/ops/test_random_mask_like.py 
157.  oneflow/python/test/ops/test_reduce_mean.py 
158.  oneflow/python/test/ops/test_reduce_ops.py
159.  oneflow/python/test/ops/test_reduce_opsV2.py 
160.  oneflow/python/test/ops/test_reduce_sum.py
161.  oneflow/python/test/ops/test_relu_fakedev.py
162.  oneflow/python/test/ops/test_reshapeV2.py
163.  oneflow/python/test/ops/test_reshapeV3.py
164.  oneflow/python/test/ops/test_scalar_by_tensor_ops.py
165.  oneflow/python/test/ops/test_shuffle.py
166.  oneflow/python/test/ops/test_sigmoid_cross_entropy.py
167.  oneflow/python/test/ops/test_smooth_l1_loss.py 
168.  oneflow/python/test/ops/test_softmax.py
169.  oneflow/python/test/ops/test_softmax_cross_entropy.py
170.  oneflow/python/test/ops/test_sort.py
171.  oneflow/python/test/ops/test_sparse_cross_entropy.py
172.  oneflow/python/test/ops/test_sparse_cross_entropy_ms.py
173.  oneflow/python/test/ops/test_sparse_softmax_cross_entropy.py
174.  oneflow/python/test/ops/test_sparse_softmax_cross_entropy_ms.py
175.  oneflow/python/test/ops/test_split_like.py
176.  oneflow/python/test/ops/test_square.py
177.  oneflow/python/test/ops/test_squeeze.py
178.  oneflow/python/test/ops/test_top_k.py
179.  oneflow/python/test/ops/test_transpose.py
180.  oneflow/python/test/ops/test_upsample.py
181.  oneflow/python/test/ops/test_util.py
182.  oneflow/python/test/ops/test_watch_diff.py


183.  oneflow/user/kernels/add_n_kernel_mlu.cpp
184.  oneflow/user/kernels/bias_add_mlu_kernel.cpp
185.  oneflow/user/kernels/concat_kernel_mlu.cpp
186.  oneflow/user/kernels/conv_kernel_cambricon.cpp
187.  oneflow/user/kernels/instance_norm_kernel_cambricon.cpp
188.  oneflow/user/kernels/normalization_kernel_mlu.cpp 
189.  oneflow/user/kernels/pool_cambricon_kernel.cpp
190.  oneflow/user/kernels/prelu_kernel_mlu.cpp 
191.  oneflow/user/kernels/relu_kernel_fakedev.cpp
192.  oneflow/user/kernels/relu_kernel_mlu.cpp
193.  oneflow/user/kernels/reshape_kernel.cpp 
194.  oneflow/user/kernels/sigmoid_kernel_mlu.cpp
195.  oneflow/user/kernels/transpose_kernel_mlu.cpp
196.  oneflow/user/kernels/upsample_kernel_mlu.cpp
197.  oneflow/user/ops/instance_norm_op.cpp
198.  oneflow/user/ops/upsample_op.cpp



