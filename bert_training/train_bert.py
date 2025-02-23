from flask import Flask, request, jsonify
import torch
import os
import pickle
import torch.optim as optim  # 使用 PyTorch 的 AdamW
import numpy as np  # 修复 inverse_transform 错误
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载 Hugging Face 模型文件并保存到 '../model'
from transformers import AutoModelForSequenceClassification

model_dir = os.path.join(os.path.dirname(__file__), '../model')
os.makedirs(model_dir, exist_ok=True)

# 下载并保存模型
print("正在下载 BERT 模型，请稍候...")
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model.save_pretrained(model_dir)

# 下载并保存 Tokenizer
print("正在下载 Tokenizer，请稍候...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer.save_pretrained(model_dir)
print("模型和 Tokenizer 下载并保存完成！")

# 加载模型和标签编码器
model = BertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
model.to(device)
model.eval()

with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)
    # 确保类别列表已排序
    label_encoders['category'].classes_ = sorted(label_encoders['category'].classes_)
    print("加载的类别列表 (已排序):", label_encoders['category'].classes_)

# 重新保存标签编码器，确保 API 使用的是排序后的版本
with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

# 加载 Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)

# AdamW 优化器 (PyTorch 版本)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 预测函数
def predict(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # 修复: 将 predicted_class 转换为 numpy 数组以兼容 inverse_transform()
    category = label_encoders['category'].inverse_transform(np.array([predicted_class]).reshape(-1))[0]
    return category

# API 路由
@app.route('/')
def index():
    return "BERT 分类 API 已启动，请使用 /predict 进行请求。"

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        text = data.get('text', None)
        if not text:
            return jsonify({"error": "Text input is required"}), 400

        category = predict(text)

        response = {
            "text": text,
            "predicted_category": category
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 启动服务 (使用端口 8000 避免冲突)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
