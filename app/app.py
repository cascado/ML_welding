import flask
from flask import render_template
import pickle
import joblib
import sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


app = flask.Flask(__name__, template_folder='template')


def predict_transform(params, columns, clf):
    list_reshape = np.array(params).reshape(1, len(params))
    x_predict = pd.DataFrame(list_reshape, columns=columns)
    x_predict_clf = clf.transform(x_predict)
    return x_predict_clf


@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    if flask.request.method == 'POST':
        with open('B:/Projects/ML/Aps/Welding/app/KNN_model_for_width.pkl', 'rb') as width:
            loaded_model_width = pickle.load(width)
        with open('B:/Projects/ML/Aps/Welding/app/DTR_model_for_depth.pkl', 'rb') as depth:
            loaded_model_depth = pickle.load(depth)
        scaler_width = joblib.load(
            'B:/Projects/ML/Aps/Welding/app/scaler_width.save')
        scaler_depth = joblib.load(
            'B:/Projects/ML/Aps/Welding/app/scaler_depth.save')
        IF = float(flask.request.form['IF'])
        VW = float(flask.request.form['VW'])
        IW = float(flask.request.form['IW'])
        FP = float(flask.request.form['FP'])
        x_transform_width_params = [IF, VW]
        x_transform_width_columns = ["IF", "VW"]
        x_transform_depth_params = [IW, VW, IF]
        x_transform_depth_columns = ["IW", "VW", "IF"]
        y_prediction_depth = loaded_model_depth.predict(predict_transform(
            x_transform_depth_params, x_transform_depth_columns, scaler_depth))
        y_prediction_width = loaded_model_width.predict(predict_transform(
            x_transform_width_params, x_transform_width_columns, scaler_width))
        return render_template('main.html', result_width=y_prediction_width, result_depth=y_prediction_depth)


if __name__ == '__main__':
    app.debug = True
    app.run()
