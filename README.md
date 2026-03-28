# Customer RFM & Churn Predictor
An end-to-end ML project using RFM Analysis for segmentation and Random Forest for churn prediction."
# 🚀 E-Commerce Customer RFM & Churn Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://customer-rfm-churn-predictor.streamlit.app/) 

### 📌 Project Overview
This project predicts customer churn for an e-retailer using a Random Forest Classifier. By analyzing over 1 million transactions, I developed a model that identifies "at-risk" customers based on their purchase behavior, allowing for proactive retention strategies.

---

### 💡"Model" Story (Technical Highlight)
This project features a critical "Pivot" that demonstrates professional data science maturity:
* **The Trap:** Initial models reached **95% accuracy** by using `Recency`. However, since churn is defined by inactivity, using recency created a **Data Leak** (predicting the past, not the future).
* **The Solution:** I deliberately removed recency-based features to build an **"Model."** * **The Result:** A robust **73.13% Accuracy** that relies on spending consistency and tenure—providing a true early-warning system.

---

### 🛠️ Tech Stack & Tools
* **Language:** Python 3.10+
* **ML Library:** Scikit-Learn (Random Forest Classifier)
* **Data Handling:** Pandas, NumPy
* **Optimization:** GridSearchCV, Cost-Function Analysis
* **Deployment:** Streamlit Cloud
* **Version Control:** Git LFS (Large File Storage)

---

### 🧪 Engineered Features
To capture "Customer Health," I engineered several custom metrics that outperformed raw data:
1. **Customer_Lifetime (Tenure):** Total days the customer has been with the brand.
2. **Purchase_Interval:** The average gap between orders (identifies "rhythm" breakers).
3. **Daily_Value:** Total spend divided by lifetime (measures daily profitability).
4. **Log_Monetary:** Normalized spending data to handle extreme outliers (whales).

---

### 📊 Model Results
| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 76.73% |
| **Testing Accuracy** | 73.13% |
| **Algorithm** | Random Forest (max_depth=8) |
| **Dataset Size** | 1,000,000+ Rows |

---

### 🏃 How to Use
1. **Live Demo:** Click the Streamlit badge at the top to test the model in your browser.
2. **Local Setup:**
   ```bash
   git clone [https://github.com/Nishant-0111/Customer-RFM-Churn-Predictor.git](https://github.com/Nishant-0111/Customer-RFM-Churn-Predictor.git)
   cd Customer-RFM-Churn-Predictor
   pip install -r requirements.txt
   streamlit run app.py
