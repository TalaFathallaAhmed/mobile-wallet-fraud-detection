import gradio as gr
import joblib
import pandas as pd

# Load the saved model and feature columns
model = joblib.load('fraud_detection_model.pkl')
columns = joblib.load('model_columns.pkl')

def predict_fraud(amount, oldbalanceOrg, newbalanceOrig,
                  oldbalanceDest, newbalanceDest, transaction_type):

    # Calculate engineered features
    errorOrig = oldbalanceOrg - amount - newbalanceOrig
    errorDest = oldbalanceDest + amount - newbalanceDest

    # Create input using the same features as training
    input_data = pd.DataFrame(0, index=[0], columns=columns)

    input_data['amount'] = amount
    input_data['oldbalanceOrg'] = oldbalanceOrg
    input_data['newbalanceOrig'] = newbalanceOrig
    input_data['oldbalanceDest'] = oldbalanceDest
    input_data['newbalanceDest'] = newbalanceDest
    input_data['errorOrig'] = errorOrig
    input_data['errorDest'] = errorDest
    input_data['step'] = 1

    # Set transaction type
    type_column = 'type_' + transaction_type
    if type_column in input_data.columns:
        input_data[type_column] = 1

    # Predict fraud probability
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.5:
        return f"⚠️ FRAUD DETECTED! Fraud probability: {probability:.2%}"

    return f"✅ LEGITIMATE TRANSACTION. Fraud probability: {probability:.2%}"


# Create the interface
app = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount"),
        gr.Number(label="Original Account Balance Before"),
        gr.Number(label="Original Account Balance After"),
        gr.Number(label="Destination Account Balance Before"),
        gr.Number(label="Destination Account Balance After"),
        gr.Dropdown(
            choices=["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
            label="Transaction Type"
        )
    ],
    outputs=gr.Textbox(label="Fraud Detection Result"),
    title="Mobile Wallet Fraud Detection",
    description="Enter transaction details to detect potentially fraudulent activity."
)

app.launch(share=True)
