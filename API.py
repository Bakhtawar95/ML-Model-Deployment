import pandas as pd
from flask import Flask, request, render_template
import mlflow

app = Flask(__name__)

model_name="stress-level"
stage="Production"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    X = pd.DataFrame([features], columns =['respiration_rate','body_temperature', 'blood_oxygen', 'heart_rate', 'sleeping_hours'], dtype = float)
    prediction=model.predict(X)
    output=prediction[0]
    return render_template('index.html', prediction_text='The predicted stress level of the human is {}'.format(output))



#if __name__ == "__main__":
    #app.run(debug=True)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8080")