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
13. 
14. oneflow/core/device/cambricon_device_stream_index.h
15. oneflow/core/device/cambricon_device_stream_index.cpp
16. 
17. oneflow/core/device/cambricon_queue_handle.h 
18. oneflow/core/device/cambricon_queue_handle.cpp
19. 
20. oneflow/core/device/device_context.h
21. 
22. oneflow/core/device/fake_device_device_context.h
23. oneflow/core/device/fake_device_device_context.cpp
24. 
25. oneflow/core/device/fake_device_stream_index.h
26. oneflow/core/device/fake_device_stream_index.cpp
27. 
28. 
