#!/usr/bin/env bash
# Usage:
# python demo.py  -c "path/to/config.yaml" -m "path/to/model.pth" --image_dir "path/to/image_dir"

# 配置文件
config_file="data/pretrained/resnet18_1.0_CrossEntropyLoss_20220822153756/config.yaml"
# 模型文件
model_file="data/pretrained/resnet18_1.0_CrossEntropyLoss_20220822153756/model/best_model_043_92.5587.pth"
# 待测试图片目录
image_dir="data/test_images/rubbish"
python demo.py \
    -c $config_file \
    -m $model_file \
    --image_dir $image_dir