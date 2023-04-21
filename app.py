from flask import Flask,render_template,request
import pickle
import numpy as np
import xgboost as xg

model = pickle.load(open('/Users/mitanshpatel/Downloads/xg.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])

def Gold_price():
    SPX = (request.form.get('SPX'))
    USO = (request.form.get('USO'))
    SLV = (request.form.get('SLV'))
    EURUSD = (request.form.get('EUR/USD'))


# prediction
    result = model.predict(np.array([SPX,USO,SLV,EURUSD]).reshape(1,4))

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(port=8500,debug=True)


