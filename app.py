
from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("pongphon.pkl")

@app.route('/api/pongphon', methods=['POST'])
def house():
    gender = int(request.form.get('gender')) 
    emoglobin = float(request.form.get('emoglobin')) 
    mch = float(request.form.get('mch')) 
    mchc = float(request.form.get('mchc')) 
    mcv = float(request.form.get('mcv')) 
    
    # Prepare the input for the model
    x = np.array([[gender, emoglobin, mch, mchc, mcv]])

    # Predict using the model
    prediction = model.predict(x)
    if(prediction==0):
        data = "เป็นโรคผอมค้าบพี่"
    elif(prediction==1):
        data = "ไม่เป็นค้าบพี่"

    # Return the result
    return {'result': data}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
