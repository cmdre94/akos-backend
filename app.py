import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from flask import (Flask, redirect, render_template, request, send_from_directory, url_for)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['GET'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

try:
    USAhousing = pd.read_csv('USA_Housing.csv')
    X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]
    y = USAhousing['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    lm = LinearRegression()
    fit = lm.fit(X_train,y_train)
    coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
    predictions = lm.predict(X_test)
except Exception as e:
    print(e)

@app.route('/getDetails', methods=['POST'])
def getDetails():
    try:
        request_data = request.get_json()
        if (request):
            housingInput = pd.DataFrame(request_data['data']['data'])
            print(housingInput)
            pricePrediction = lm.predict(housingInput)
            details = {
                "flowStatus": "SUCCESS",
                "flowStatusMessage": "",
                "result": {
                    "intercept": lm.intercept_,
                    "coefficients": coeff_df.to_json(),
                    "prediction": pricePrediction[0],
                    "metrics": {
                        'mse': metrics.mean_squared_error(y_test, predictions),
                        'mae': metrics.mean_absolute_error(y_test, predictions),
                        'rmse': np.sqrt(metrics.mean_squared_error(y_test, predictions))
                    }
                }
            }
            return details
        else:
            return {
                "flowStatus": "FAILURE",
                "flowStatusMessage": "Backend Error!",
                "result": {},
                "error": "error"
            }
    except Exception as e:
        return {
            "flowStatus": "FAILURE",
            "flowStatusMessage": str(e),
            "result": {}
        }

if __name__ == '__main__':
   app.run(port=8000, debug=True)
