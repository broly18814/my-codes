import pickletools
from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

## import the ridge and scaler pkl

# Ensure your 'rd.pkl' and 'scaler.pkl' files are in the same directory, 
# or provide the correct path.
rd=pickle.load(open('rd.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))



@app.route('/')
def index():
    # Assuming your main form HTML is in index.html
    return render_template('index.html')


# Corrected: Changed route to match the url_for('predict_datapoint') in the HTML
@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        # NOTE: StandardScaler expects a 2D array, which you've correctly provided.
        # It's good practice to wrap in a DataFrame if the scaler was fitted with feature names.
        
        new_data = [[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]
        new_data_scaled = scaler.transform(new_data)
        result = rd.predict(new_data_scaled)

        # Corrected: Rendering 'index.html' (your form page) and passing the result as 'result'
        # to match the Jinja check {% if result is defined %}
        return render_template('home.html',result=result[0])

    else:
        # Corrected: Rendering 'index.html' for a GET request
        return render_template('home.html')



if __name__ == "__main__":
    # Use debug=True for easier development, though not strictly required for this fix
    app.run(host="0.0.0.0",port=8080)