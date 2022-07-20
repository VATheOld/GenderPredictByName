from flask import Flask, render_template, request
import pickle

import SVM_model


def name_cleaning1(Name):
    Name = Name.split(" ")
    middleName = Name[1:len(Name) - 1]
    firstName = Name[len(Name) - 1:]
    name = middleName + firstName
    inputName = " ".join(name)
    inputName = inputName.lower()
    return inputName

app = Flask(__name__)

model1 = pickle.load(open('SVM_model.pkl','rb'))

@app.route('/',methods=['GET'])
def index_page():
    return render_template('indexViet.html')

@app.route('/', methods=['POST'])
def predictVietnamese():
    if request.method == 'POST':
        Name = str(request.form.get('Vietnamese name'))
        inputName = name_cleaning1(Name)
        pred_name = SVM_model.makePrediction(inputName)
    if pred_name == '0':
        pred = 'Male'
    elif pred_name== '1':
        pred = 'Female'
    result=[Name,pred]
    return render_template('indexViet.html', output=result)

if __name__ == '__main__':
    app.run(debug = True)
