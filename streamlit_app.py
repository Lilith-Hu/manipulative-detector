import streamlit as st
import requests

# 🌍 语言选择
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"])

# 🌟 设置标题 & 介绍文本
st.title("🚀 Unwanted Message Analyzer")
if lang == "English":
    st.write(
        "This tool helps classify messages into different categories such as harassment, threats, or neutral messages. Simply enter your text below and get an instant classification!")
else:
    st.write("这个工具可以帮助分类消息，比如骚扰、威胁或中性消息。输入文本，立即获取分类结果！")

st.markdown("---")

# 🎨 **优化界面**
st.markdown(
    """
    <style>
        .reportview-container {
            background: #f4f4f4;
            padding: 20px;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        .stTextInput {
            border-radius: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🎯 **用户输入**
user_text = st.text_area("📩 Enter a message / 输入消息", height=100)

# 🌐 **API 地址**
API_URL = "https://unwanted-message-api-2-0.onrender.com/predict"

# 🚀 **提交按钮**
if st.button("Analyze / 分析"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text! / 请输入文本！")
    else:
        try:
            # 发送请求
            response = requests.post(API_URL, json={"text": user_text})
            if response.status_code == 200:
                data = response.json()
                category = data.get("predicted_category", "Unknown")

                # 🎯 **显示结果**
                st.success(f"🎯 **Category / 分类**: {category}")
            else:
                st.error("❌ API returned an error! Please try again later. / API 出错，请稍后重试。")

        except requests.exceptions.RequestException:
            st.error("❌ Failed to connect to API. / 无法连接到服务器，请检查网络。")

