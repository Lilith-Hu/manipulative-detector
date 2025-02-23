from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import pickle
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
CORS(app)  # 添加这行


# 设置设备
device = torch.device("cpu")

# 加载模型和标签编码器
model_dir = os.path.join(os.path.dirname(__file__), 'model')

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=12)
model.load_state_dict(torch.load(os.path.join(model_dir, 'bert_model.pt'), map_location=device))
model.to(device)
model.eval()

with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)

# 加载 Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)

# 预测函数
####

#########
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
            # 转换为 numpy array
            predicted_class = predicted_class.cpu().numpy()

        # 转换预测结果为类别标签
        category = label_encoders['category'].classes_[predicted_class[0]]
        return category

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise e
#########

# API 路由
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

# 启动服务


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)