import pandas as pd 
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import sqlite3



import xgboost
import joblib


class F1Optimizer():
    def __init__(self):                                             # https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

def timeit(P):
    s = datetime.now()
    F1Optimizer.maximize_expectation(P)
    e = datetime.now()
    return (e-s).microseconds / 1E6



def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
#         if feature_name in range(10):
#             continue
        mean = df[feature_name].mean()
        std = df[feature_name].std()
       # max = df[feature_name].max()
        #min = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min) / max-min
        result[feature_name] = (df[feature_name] - mean) / std
    return result


   
    
from flask import Flask, jsonify, request

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])

def deployment():
    
    dict1={}
    dict2={}
    df=pd.read_csv('order_products__train.csv')
    df1=pd.read_csv('orders.csv')
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())

    to_predict_list=[int(i) for i in to_predict_list[0].split()]
    order=pd.DataFrame(to_predict_list,columns=['order_id'])
#     features=pd.read_csv('features_1.csv')
#     features.pop('Unnamed: 0')
#     f,na_list=reduce_mem_usage(features)
#     del features
#     gc.collect()
  

    conn = sqlite3.connect("my_data.db")
    
  
    df2=df1.merge(order,how='inner',on='order_id')
    user_id=np.unique(df2['user_id'])
    m=""
    for i in user_id:
        m=m+str(i)+','
  
    
    f=pd.read_sql_query(""" select * from features where user_id in ("""+m[:-1]+')',conn)
    x=f.merge(df2[['user_id','order_id']],how='right',on='user_id') 
    x=x.merge(df[['product_id','order_id']],how='left',on=['product_id','order_id'])
    bst= joblib.load('model.pkl')
    d_test = xgboost.DMatrix(normalize(x.drop([ 'user_id', 'order_id', 'product_id'], axis=1)))         
    x['reordered'] = (bst.predict(d_test))# > 0.21).astype(int)         # deep learning model f1 eval
    final=x[['user_id','product_id','reordered','order_id']]
    
    grps=final.groupby('user_id')
    for i,grp in tqdm(grps):
        out= F1Optimizer.maximize_expectation(grp['reordered'].sort_values(ascending=False), None)
        dict1[i]=out[0]
        
    user_id=np.unique(final['user_id'])
    glob=0
    for i in range(len(user_id)):
        df=final[final['user_id']==user_id[i]]
        df=df.sort_values('reordered',ascending=False)
        if dict1[user_id[i]]=='None' or dict1[user_id[i]]==0:
            continue
        else:
            df=df.iloc[:dict1[user_id[i]]]
        temp=df
        if i==0:
            glob=temp
        else:
            glob=pd.concat([glob,temp])
    if type(glob)==int:
        pass
        
    else:
        glob['reordered'][:]=1
    
    
   
    if type(glob)!=int:
        for row in glob.itertuples():
            try:
                if row.reordered== 1:
                    dict2[row.order_id] += ' ' + str(row.product_id)
            except:
                dict2[row.order_id] = str(row.product_id)

    for order in to_predict_list:
        if order not in dict2:
            dict2[order] = 'None'
    #sub = pd.DataFrame.from_dict(dict2, orient='index')
    #sub.reset_index(inplace=True)
    #sub.columns = ['order_id', 'products']
    return jsonify(dict2)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

