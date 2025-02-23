import streamlit as st
import requests

# 页面配置
st.set_page_config(
    page_title="Unwanted Message Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# 自定义 CSS
st.markdown("""
    <style>
        /* 主题色彩 */
        :root {
            --primary-color: #1E88E5;
            --secondary-color: #FFC107;
            --background-color: #f8f9fa;
            --text-color: #2c3e50;
        }

        /* 整体布局 */
        .main {
            padding: 2rem;
            background-color: var(--background-color);
        }

        /* 标题样式 */
        h1 {
            color: var(--primary-color);
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }

        /* 子标题样式 */
        .subtitle {
            font-family: 'Georgia', serif;
            color: #666;
            font-style: italic;
            font-size: 1.2rem;
        }

        /* 输入框样式 */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 15px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .stTextArea textarea:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(30,136,229,0.2);
        }

        /* 按钮样式 */
        .stButton button {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #1565C0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* 结果卡片样式 */
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 20px;
        }

        /* 标签样式 */
        .category-tag {
            background-color: var(--secondary-color);
            color: #fff;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    lang = st.selectbox("Language / 语言", ["English", "中文"])
    st.markdown("---")

    if lang == "English":
        st.markdown("### About")
        st.write("This tool uses advanced BERT AI to analyze and classify potentially unwanted messages.")
    else:
        st.markdown("### 关于")
        st.write("此工具使用先进的BERT人工智能技术来分析和分类潜在的不良信息。")

# 主界面
st.markdown('<h1 style="text-align: center;">🛡️ Unwanted Message Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="text-align: center;">Powered by BERT Model - 2024 Edition</p>',
            unsafe_allow_html=True)

# 介绍文本
if lang == "English":
    st.markdown("""
    > This advanced tool helps identify and classify various types of unwanted messages, including:
    - Harassment
    - Threats
    - Emotional manipulation
    - Neutral content

    Simply enter your text below for instant analysis.
    """)
else:
    st.markdown("""
    > 这个先进的工具可以帮助识别和分类各种类型的不良信息，包括：
    - 骚扰信息
    - 威胁信息
    - 情感操控
    - 中性内容

    在下方输入文本即可获得即时分析。
    """)

# 用户输入区
col1, col2 = st.columns([2, 1])
with col1:
    user_text = st.text_area(
        "📝 Enter message / 输入消息" if lang == "English" else "📝 输入消息 / Enter message",
        height=150
    )

# API 配置
API_URL = "https://unwanted-message-api-2-0.onrender.com/predict"

# 分析按钮
if st.button("🔍 Analyze" if lang == "English" else "🔍 开始分析"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text!" if lang == "English" else "⚠️ 请输入文本！")
    else:
        try:
            with st.spinner('Analyzing...' if lang == "English" else '分析中...'):
                response = requests.post(API_URL, json={"text": user_text})
                if response.status_code == 200:
                    data = response.json()
                    category = data.get("predicted_category", "Unknown")

                    # 显示结果
                    st.markdown("""
                    <div class="result-card">
                        <h3>Analysis Result</h3>
                        <p><strong>Input Text:</strong></p>
                        <p style="background: #f8f9fa; padding: 10px; border-radius: 5px;">{}</p>
                        <p><strong>Detected Category:</strong></p>
                        <span class="category-tag">{}</span>
                    </div>
                    """.format(user_text, category), unsafe_allow_html=True)
                else:
                    st.error("❌ API Error" if lang == "English" else "❌ API 错误")
        except requests.exceptions.RequestException:
            st.error("❌ Connection Error" if lang == "English" else "❌ 连接错误")

# 页脚
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>© 2024 Unwanted Message Analyzer - BERT Edition</p>",
    unsafe_allow_html=True
)