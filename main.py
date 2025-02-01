import joblib

# **📌 1️⃣ 加载已保存的模型**
svm_model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("✅ 模型、TF-IDF 向量化器 和 标签编码器 已成功加载！")

# **📌 2️⃣ 定义预测函数**
def predict_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])  # 进行 TF-IDF 处理
    pred_label = svm_model.predict(text_tfidf)  # 预测类别
    pred_category = label_encoder.inverse_transform(pred_label)  # 转换回原标签

    print("\n🌟 **预测结果** 🌟")
    print(f"类别: {pred_category[0]}")
    print("==================================================")

# **📌 3️⃣ 进入交互模式**
while True:
    user_input = input("\n输入一句话进行分类（输入 'exit' 退出）：")
    if user_input.lower() == "exit":
        break
    predict_text(user_input)
