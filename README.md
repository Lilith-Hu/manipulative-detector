# ThesisTool_BERT

## 项目描述
ThesisTool_BERT 是一个基于 BERT 的文本分类工具，旨在进行高效准确的文本分类任务。

## 模型下载
模型已托管在 Hugging Face：[BERT 分类模型](https://huggingface.co/LilithHu/bert-classifier)

### 下载步骤
1. 访问 Hugging Face 模型页面
2. 点击 "Files and versions" 下载模型文件
3. 将下载的文件放置在项目的 `model/` 目录下

## 使用说明
### 环境配置
1. 克隆仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 配置模型路径

### 快速开始
```python
from model import BERTClassifier

# 加载模型
model = BERTClassifier.load_from_huggingface()

# 进行文本分类
result = model.predict("your text here")