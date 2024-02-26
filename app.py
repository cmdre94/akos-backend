import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import string
import json
# import mysql.connector

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, \
     roc_auc_score, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as mt
from sklearn.feature_selection import SelectKBest, chi2


app = Flask(__name__)

try:
   print("Python backend started")
except Exception as e:
   print(e)

@app.route('/')
def index():
   return 'Hello World'

@app.route('/hello')
def hello():
   return {"message": "hello world"}

@app.route('/getDetails', methods=['POST'])
def getDetails():
   try:
      request_data = request.get_json()
      # request_df = pd.json_normalize(request_data)
      response = {"flowStatus": "SUCCESS", "flowStatusMessage": "", "results": request_data}
      return response
   except Exception as e:
      return {"flowstatus": "FAILURE", "flowStatusMessage": e, "results": {}}

if __name__ == '__main__':
   app.run()
