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

@app.route('/', methods=['GET'])
def hello():
    try:
        print('Request for hello response received')
        response = {
            "flowstatus": "SUCCESS",
            "flowStatusMessage": "Request complete",
            "result": "Hello World!"
            }
        return response
    
    except Exception as e:
        return {
            "flowStatus": "FAILURE",
            "flowStatusMessage": str(e),
            "result": {}
        }

if __name__ == '__main__':
   app.run(port=8000, debug=True)
