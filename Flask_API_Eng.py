from flask import Flask, render_template, request
import DecisionTree_model

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index_page():
    return render_template('indexEng.html')

def name_cleaning(Name):
    Name = Name.split(" ")
    for component in range(len(Name)):
        firstName = Name[0]
    return firstName

def features(name):
    name = str(name).lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letter'
        's': name[-3:],
    }
def engNameGenderPredict(name):
    n=[name]
    result=DecisionTree_model.makePredictionEng(n)
    return result

@app.route('/', methods=['POST'])
def predictEnglish():
    if request.method == 'POST':
        Name = str(request.form.get('English name'))
        inputName = name_cleaning(Name)
        pred_name = engNameGenderPredict(inputName)
    if pred_name == 0:
        pred = 'Male'
    elif pred_name == 1:
        pred = 'Female'
    result = [Name, pred]
    return render_template('indexEng.html', output=result)

if __name__ == '__main__':
    app.run(debug = True)