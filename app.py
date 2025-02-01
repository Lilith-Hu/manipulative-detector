from flask import Flask, request, jsonify, Response
import joblib
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# **📌 1️⃣ 加载已训练的 SVM 模型 & TF-IDF 向量化器**
svm_model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# **📌 2️⃣ 初始化 Flask**
app = Flask(__name__)

# **📌 3️⃣ 文本预处理**
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 去除多余空格
    return text.strip().lower()  # 统一小写 & 去除前后空格

# **📌 4️⃣ 预测类别函数**
def predict_category(text):
    text = clean_text(text)  # 预处理文本
    text_tfidf = tfidf_vectorizer.transform([text])  # 转换为 TF-IDF 向量
    pred_label = svm_model.predict(text_tfidf)  # 预测类别
    pred_category = label_encoder.inverse_transform(pred_label)  # 转换回原标签
    return pred_category[0]

# **📌 5️⃣ 首页 API**
@app.route("/")
def home():
    return "🚀 Flask API 运行成功！使用 /predict 端点进行文本分类。"

# **📌 6️⃣ 预测 API**
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # 读取 JSON 请求

    # **✅ 处理输入为空的情况**
    if not data or "text" not in data:
        return jsonify({"error": "请提供文本"}), 400

    text = data["text"]
    category = predict_category(text)

    # **✅ 使用 `json.dumps()` 确保中文正常显示**
    response_data = json.dumps(
        {"text": text, "predicted_category": category}, ensure_ascii=False
    )

    # **✅ `Response()` 确保 UTF-8 编码**
    return Response(response_data, content_type="application/json; charset=utf-8")

# **📌 7️⃣ 运行 Flask 服务器**
if __name__ == "__main__":
    app.run(debug=True, port=5001)  # ✅ 使用 5001 避免端口冲突
