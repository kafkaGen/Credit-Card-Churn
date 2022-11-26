import json
import pandas as pd
import numpy as np
from flask import render_template, redirect, request, jsonify, make_response, flash
from yellowbrick.target.feature_correlation import feature_correlation

from app import app
from app.forms import DataForm
from utils import DataLoader
from utils import Predictor
from utils import Dataset
from settings.constants import TRAIN_CSV, VAL_CSV


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = DataForm()

    if form.validate_on_submit():
        train = pd.read_csv(TRAIN_CSV)
        values = [v for k, v in request.form.items() if k in train.columns]
        keys = [k for k, v in request.form.items() if k in train.columns]

        df = pd.DataFrame([values], columns=keys)
        number = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                  'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                  'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        df[number] = df[number].astype(np.float)

        dl = DataLoader()
        dl.fit(train)
        dl.load_data()
        df_pre = dl.one_transform(df.copy())

        predictor = Predictor()
        prediction = predictor.predict(df_pre)[0]
        probability = predictor.predict_proba(df_pre)[0]

        if prediction:
            message = f"Well, that's okay, there's a {probability[1]:0.5f} percent chance that this customer will stay with you."
        else:
            message = f'Oops! Time to pay attention to this client. There is a {probability[0]:0.5f} percent chance that you will lose him soon.'

        return render_template('predict.html', title='Predict', form=form, messages=[message])

    return render_template('predict.html', title='Predict', form=form)


@app.route('/dataset')
def dataset():
    df = pd.read_csv(TRAIN_CSV)
    df_n = df.sample(10)
    return render_template('dataset.html', title='Dataset', tables=[df_n.to_html(classes='data')], titles=df.columns.values)


@app.route('/perfomance', methods=['GET', 'POST'])
def perfomance():
    return render_template('perfomance.html', title='Perfomance')
