# styletransform

## 使用方法

- 从`https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/styletranform.tar.gz`下载模型和图片到当前`infer.sh`所在的目录
- 执行`sh infer.sh`即可获得StyleNet风格化（素描）后的结果图片

## 保存 model（for serving）

首先下载并解压模型所需的参数

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cambricon/styletranform.tar.gz
tar zxvf styletranform.tar.gz
```

如果想导出 gpu 平台的模型，则要在支持 gpu 的环境中，执行
```bash
bash save_style_model.sh gpu
```

如果想导出 寒武纪 平台的模型，则要在支持 寒武纪 的环境中，执行
```bash
bash save_style_model.sh cambricon
```

save_style_model.sh 中的参数
- backend: 设置 device 类型
- model_dir: 模型所需的参数的目录，通过上面的下载解压命令后可以得到的 stylenet_nhwc 目录即是参数目录
- save_dir: 模型保存的目录
- model_version: 保存模型的版本号
- image_width: 模型兼容的输入 image 的 width
- image_height: 模型兼容的输入 image 的 height
- force_save: 当 model_version 所指定的保存的模型的版本号已存在时，是否强制覆盖保存