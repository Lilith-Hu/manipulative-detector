import streamlit as st
import requests

# 页面配置
st.set_page_config(
    page_title="Manipulative Language Detector",
    page_icon="🧠",
    layout="wide"
)

# 侧边栏语言选择
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"])
st.sidebar.markdown("---")

# 标题
if lang == "English":
    st.title("🧠 Manipulative Language Detector")
    st.markdown("This tool uses an AI model to detect manipulative language in messages.")
else:
    st.title("🧠 情感操控语言识别器")
    st.markdown("本工具使用 AI 模型检测文本中的情感操控语言。")

st.markdown("---")

# 用户输入
user_text = st.text_area("Enter your message / 输入文本", height=150)

# 后端 API 地址
API_URL = "https://manipulative-detector-api.onrender.com"


# 分析按钮
if st.button("🔍 Analyze / 分析"):
    if not user_text.strip():
        st.warning("⚠️ Please enter a message!" if lang == "English" else "⚠️ 请输入内容！")
    else:
        try:
            with st.spinner("Analyzing..." if lang == "English" else "分析中..."):
                response = requests.post(API_URL, json={"text": user_text})
                if response.status_code == 200:
                    result = response.json().get("predicted_category", "")
                    if result == "manipulative":
                        st.error("⚠️ Manipulative Language Detected!" if lang == "English" else "⚠️ 检测到操纵性语言！")
                    else:
                        st.success("✅ No manipulative language detected." if lang == "English" else "✅ 未检测到操纵性语言。")
                else:
                    st.error("❌ API returned an error.")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

# 页脚
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>© 2024 Manipulative Language Detector</p>",
    unsafe_allow_html=True
)

