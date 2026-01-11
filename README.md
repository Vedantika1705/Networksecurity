# Network Security ML Project 🚀

This project is a **Machine Learning application for network security**.  
It predicts phishing or malicious URLs using a trained ML model and provides a **web interface via FastAPI**.

---

## 🧠 What It Does
- Ingests CSV files containing network/security data
- Preprocesses the data automatically
- Runs a trained ML model to predict suspicious or safe URLs
- Returns predictions both as a CSV and as an **HTML table**
- Saves **example predictions** and **table images** for visualization

---

## 📊 Features
- ## 📸 Screenshots / Demo

### 🔹 Prediction Output Table
This table is automatically generated after CSV upload and prediction.

![Prediction Table](images/prediction_table.png)

### 🔹 Prediction Output Files
- **Full predictions:** [`prediction_output/output.csv`](prediction_output/output.csv)
- **Sample output (first 10 rows):** [`prediction_output/example_output.csv`](prediction_output/example_output.csv)


---

## ⚠️ Model Limitations
- Trained on a specific phishing dataset; accuracy may vary on new/unseen data
- Columns in CSV must match the training data (extra columns like `Result` are dropped automatically)
- Focus is on demonstrating **ML deployment and prediction workflow**, not achieving production-grade detection

---

## 🛠 Tech Stack
- Python
- FastAPI
- Pandas
- Scikit-learn
- Jinja2 templates
- MongoDB

---

## 🚀 How to Run

## 1️⃣ Clone the repository
```bash
git clone https://github.com/Vedantika1705/Networksecurity.git
cd Networksecurity

```
## 2️⃣ Install dependencies
pip install -r requirements.txt

## 3️⃣ Start FastAPI app
python app.py

## 4️⃣ Open in browser
 Go to http://127.0.0.1:8000 → home page
 Go to http://127.0.0.1:8000/docs → OpenAPI docs for API testing

## 5️⃣ Upload CSV
 Use /predict route in docs or via the home page form
 Example CSV: phisingData.csv

## 6️⃣ Output
 Full predictions: prediction_output/output.csv
 Example predictions: prediction_output/example_output.csv
 Table image: images/prediction_table.png

## 🔍 Demo Screenshots
 Prediction Table Example (images/prediction_table.png)
 Example Prediction CSV (prediction_output/example_output.csv) contains the first 10 predictions





