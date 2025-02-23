import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# 加载 BERT Tokenizer 和模型
MODEL_PATH = "bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 检测 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载标签编码器
label_encoder = joblib.load("label_encoder.pkl")

# 定义推理函数
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred_label_id = torch.argmax(logits, dim=1).item()
    pred_category = label_encoder.inverse_transform([pred_label_id])[0]

    return pred_category

if __name__ == "__main__":
    while True:
        user_input = input("\n输入一句话进行分类（输入 'exit' 退出）：")
        if user_input.lower() == "exit":
            break
        result = predict_text(user_input)
        print(f"预测类别: {result}")


