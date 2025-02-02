import streamlit as st
import requests

# Streamlit 页面标题
st.title("📢 Unwanted Message Analyzer")
st.subheader("输入一段文本，让 AI 预测其类别")

# 用户输入框
user_input = st.text_area("请输入文本进行分析:")

# 提交按钮
if st.button("分析文本"):
    if user_input.strip():
        # Flask API 的 URL（你的 Render API 地址）
        api_url = "https://unwanted-message-api-2-0.onrender.com/predict"


        # 发送 POST 请求到 Flask API
        response = requests.post(api_url, json={"text": user_input})

        if response.status_code == 200:
            result = response.json()
            st.success(f"🔍 预测类别: **{result['predicted_category']}**")
        else:
            st.error("❌ API 访问失败，请检查服务器状态")
    else:
        st.warning("⚠️ 请输入文本后再点击分析")

# 版权信息
st.markdown("---")
st.caption("🚀 Powered by Lilith-Hu | Streamlit & Flask API")
