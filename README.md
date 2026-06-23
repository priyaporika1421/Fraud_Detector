📌 Project Overview

This project is an AI-based Fraud Detection System that detects suspicious mobile transactions using Machine Learning. It generates synthetic transaction data, trains a model, and predicts whether a transaction is Safe, Medium Risk, or High Risk.

🚀 Features
🔹 Synthetic mobile transaction data generation
🔹 Feature engineering for fraud patterns
🔹 Machine Learning model (Random Forest)
🔹 Fraud probability prediction
🔹 Risk classification (LOW / MEDIUM / HIGH)
🔹 Streamlit web app interface (optional)
🧠 Tech Stack
Python
Pandas, NumPy
Scikit-learn
Streamlit (for UI)
📁 Project Structure
fraud_Detector/
│
├── Fraud_Detector.py     # Core ML logic
├── main.py               # Console runner
├── app.py                # Streamlit web app
├── README.md             # Project documentation
⚙️ Installation
1. Clone or open project folder
cd C:\Users\Hp\fraud_Detector
2. Install dependencies
pip install numpy pandas scikit-learn streamlit
▶️ How to Run (Console Version)
python main.py

You will see output like:

✔ Model trained successfully
{'fraud_probability': 39.0, 'risk_level': 'MEDIUM'}
🌐 How to Run (Web App - Streamlit)
python -m streamlit run app.py

Then open:

http://localhost:8501
📊 Output Example
Fraud Probability: 85% → HIGH RISK 🚨
Fraud Probability: 40% → MEDIUM ⚠
Fraud Probability: 10% → LOW SAFE ✅
🎯 Use Cases
Banking fraud detection
Mobile payment security
Account takeover detection
Behavioral anomaly detection
📌 Future Improvements
Real-time transaction API (FastAPI)
Dashboard with graphs
Deep learning model (LSTM)
Deployment on cloud (Render / AWS)
👨‍💻 Author

Student Project – AI/ML Fraud Detection System
