# 🏍️ Motorcycle Price Predictor (Sri Lanka)

A **Streamlit-based web application** that predicts the market price of **used motorcycles in Sri Lanka** using an **Ensemble Machine Learning Model** trained on locally scraped data.

---

## 🚀 Live App
👉 **[Launch the App](https://motorcycle-price-predictor-sri-lanka.streamlit.app)**

---

## 🧠 Project Overview

This app estimates motorcycle prices in Sri Lanka based on key features such as:

- Brand & Model  
- Engine Capacity (cc)  
- Year of Manufacture  
- Mileage  
- Condition  
- Start Type  

The prediction is powered by a **custom ensemble model** that averages predictions from multiple algorithms (Random Forest, Extra Trees, Gradient Boosting, XGBoost) to provide robust and reliable estimates.

---

## 📊 Dataset Source

Data was collected via **web scraping** from:
- [Riyasewana.lk](https://riyasewana.com)  
- [Ikman.lk](https://ikman.lk)

This dataset includes real Sri Lankan market listings for popular models such as:
- Bajaj CT-100  
- Bajaj Discover 125  
- Honda DIO  
- TVS Ntorq 125  
- Yamaha FZ Version 2  
- Yamaha Ray ZR  

---

## 🧩 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| **Frontend / App** | [Streamlit](https://streamlit.io) |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Data Handling** | Pandas, NumPy |
| **Model Serialization** | Joblib, Git LFS |
| **Deployment** | Streamlit Cloud |

---

## ⚙️ Local Setup

If you want to run this project locally:

```bash
# 1️⃣ Clone the repository
git clone https://github.com/RavinduKavinda22/Motorcycle-Price-Predictor-App.git

# 2️⃣ Move into the project directory
cd Motorcycle-Price-Predictor-App

# 3️⃣ Create and activate virtual environment (Mac)
python3 -m venv .venv
source .venv/bin/activate

# 4️⃣ Install dependencies
pip install -r requirements.txt

# 5️⃣ Run the app
streamlit run app.py
