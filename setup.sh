#!/bin/bash

echo "========================================"
echo "🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG STREAMLIT"
echo "========================================"

# 1) Tạo virtual environment
echo "🔧 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2) Update pip
echo "⬆ Updating pip..."
pip install --upgrade pip

# 3) Cài đặt các thư viện từ requirements.txt
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4) Cài thêm NLTK stopwords nếu cần
echo "📚 Installing NLTK stopwords..."
python3 - <<EOF
import nltk
nltk.download('stopwords')
EOF

# 5) Tạo thư mục nếu thiếu
echo "📂 Checking project folders..."
mkdir -p images
mkdir -p models_final_project_1_bai2
mkdir -p Data

# 6) In thông báo hoàn tất
echo "========================================"
echo "🎉 INSTALL FINISHED — READY TO RUN"
echo "========================================"

# 7) Chạy Streamlit
echo "🚀 Starting Streamlit app..."
streamlit run app.py
