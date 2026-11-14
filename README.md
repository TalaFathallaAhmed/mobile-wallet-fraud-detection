# Mobile Wallet Fraud Detection

## Project Overview
This project uses machine learning to identify potentially fraudulent mobile wallet transactions. It uses the PaySim synthetic financial transaction dataset.

## Dataset
- Dataset: PaySim
- Original records: 6,362,620
- Features: Transaction amount, transaction type, account balances, and engineered balance-error features.
- Target variable: `isFraud`
- Fraud cases in the original dataset: 8,213

A sample of 1,000,000 transactions was used for model development.

## Machine Learning Models
Three classification models were trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

## Model Results

| Model | Precision | Recall | F1 Score | Average Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0239 | 0.9486 | 0.0467 | 0.5722 |
| Decision Tree | 0.9547 | 1.0000 | 0.9768 | 0.9995 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Results are from the held-out test sample at a 0.5 classification threshold. Further validation is needed before real-world use.

## Interactive Application
A Gradio application allows users to enter transaction details and receive a fraud prediction with a probability score.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Gradio
- Joblib
- Google Colab

## Project Files
- `fraud_detection_model.pkl` – Trained Random Forest model
- `model_columns.pkl` – Model input column names
- `app.py` – Gradio application
- `.ipynb` – Project notebook
- `.pptx` – Project presentation

## Limitations
The dataset is synthetic and may not represent real-world fraud patterns. The high model scores require further investigation and validation. The application is a demonstration, not a production financial security system.

## Author
[Tala Fathalla Ahmed]
