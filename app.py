# ✅ 修改后的 app.py


from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 设置设备
device = torch.device("cpu")

# ========================= 🟠 加载模型 =========================
print("Downloading model from Hugging Face...")
model = AutoModelForSequenceClassification.from_pretrained("LilithHu/mbert-manipulative-detector")
tokenizer = AutoTokenizer.from_pretrained("LilithHu/mbert-manipulative-detector")
model.to(device)
model.eval()
print("Model and tokenizer loaded successfully!")

# ========================= 🟣 预测函数 =========================
def predict(text):
    try:
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        return predicted_class  # 返回 0 或 1

    except Exception as e:
        print(f"Prediction error: {e}")
        raise e

# ========================= 🔵 API 路由 =========================
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        text = data.get('text', None)
        if not text:
            return jsonify({"error": "Text input is required"}), 400

        prediction = predict(text)
        label = "操纵性语言" if prediction == 1 else "非操纵性语言"

        return jsonify({
            "text": text,
            "prediction": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================= 🟡 启动服务 =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port)
