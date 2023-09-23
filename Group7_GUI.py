#importing necessary packages
from flask import Flask, render_template, request
import pickle
import numpy as np
from Group7_Source_Program import DeepNeuralNetwork, KNN

#importing the models
indices = pickle.load(open('indices.pkl', 'rb'))
model_ANN = pickle.load(open('mushroom_ANN.pkl', 'rb'))
model_KNN = pickle.load(open('mushroom_KNN.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('Group7_GUI_Home.html')

#getting the data from input
@app.route('/predict', methods=['POST'])
def home():
    model_type = request.form['Model']
    dataa = request.form['a']
    datab = request.form['b']
    datac = request.form['c']
    datad = request.form['d']
    datae = request.form['e']
    dataf = request.form['f']
    datag = request.form['g']
    datah = request.form['h']
    datai = request.form['i']
    dataj = request.form['j']
    datak = request.form['k']
    datal = request.form['l']
    datam = request.form['m']
    datan = request.form['n']
    datao = request.form['o']
    datap = request.form['p']
    dataq = request.form['q']
    datar = request.form['r']
    datas = request.form['s']
    datat = request.form['t']
    datau = request.form['u']
    arr = np.array([dataa, datab, datac, datad, datae, dataf, datag, datah, datai, dataj, datak, datal,
                    datam, datan, datao, datap, dataq, datar, datas, datat, datau])
    
    #normalization and label encoding
    encoded_lab = []
    for ind, att in enumerate(list(indices)[1:]):
        encoded_lab.append(indices[att][arr[ind]])
    
    #selecting the model based on the input
    if model_type == "ANN":
        pred = model_ANN.predict(np.array([encoded_lab]))
    else:
        pred = model_KNN.predict(np.array([encoded_lab]))
    return render_template('Group7_GUI_Second_page.html', data=pred[0])


if __name__ == "__main__":
    app.run(debug=True)