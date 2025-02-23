from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import pickle
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 设置设备
device = torch.device("cpu")

# ========================= 🟠 加载模型 =========================
print("Downloading model from Hugging Face...")

# 从 Hugging Face Hub 下载模型权重 (bert_model.pt)
model_path = hf_hub_download(
    repo_id="LilithHu/bert-classifier",
    filename="bert_model.pt",
    local_dir="./model",  # 下载到本地 model 文件夹，防止重复下载
    force_download=False  # 设置为 False，避免每次重新下载
)

# 初始化 Bert 模型
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=12)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# ========================= 🟡 加载标签编码器 =========================
# 加载 label_encoder.pkl（假设它仍然位于本地 model 文件夹中）
print("Loading label encoder...")
model_dir = os.path.join(os.path.dirname(__file__), 'model')
with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)
print("Label encoder loaded successfully!")

# ========================= 🟢 加载 Tokenizer =========================
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # 使用公开 tokenizer
print("Tokenizer loaded successfully!")


# ========================= 🟣 预测函数 =========================
def predict(text):
    try:
        # Tokenize input
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1)
            predicted_class = predicted_class.cpu().numpy()

        # 转换预测结果为类别标签
        category = label_encoders['category'].classes_[predicted_class[0]]
        return category

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise e


# ========================= 🔵 API 路由 =========================
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


# ========================= 🟡 启动服务 =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port)

