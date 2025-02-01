import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# **📌 1️⃣ 读取 CSV 数据**
file_path = "纯中文数据库.csv"
df = pd.read_csv(file_path, sep=";")

# **📌 2️⃣ 处理文本**
df["text"] = df["text"].astype(str)

# **📌 3️⃣ 处理类别**
df["category"] = df["category"].str.strip().str.lower()
df["category"] = df["category"].apply(lambda x: re.sub(r"\s+", "-", x))

# **📌 4️⃣ 纠正拼写错误**
corrections = {
    "harassement": "harassment",
    "harassemnt": "harassment",
    "delusional--speech": "delusional-speech",
}
df["category"] = df["category"].replace(corrections)

# **📌 5️⃣ 进行 TF-IDF 处理**
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["text"])

# **📌 6️⃣ 进行标签编码**
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["category"])

# **📌 7️⃣ 拆分数据集**
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# **📌 8️⃣ 训练 SVM 模型**
svm_model = SVC(kernel="linear", class_weight="balanced", random_state=42)
svm_model.fit(X_train, y_train)

# **📌 9️⃣ 保存模型文件**
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ SVM 模型、TF-IDF 向量化器 和 标签编码器 已保存！")
