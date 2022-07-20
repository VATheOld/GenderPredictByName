import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Train the classification model
col = ['Name', 'Gender']
VietNamelist = pd.read_csv("Vietnamese_Name_gender.csv", usecols=col)
VietNamelist.Gender.replace({"M": 0, "F": 1}, inplace=True)

VietNamelist['Name'] = VietNamelist['Name'].str.lower()

Xfeature=VietNamelist['Name']
cv=CountVectorizer()
X=cv.fit_transform(VietNamelist['Name'].values.astype(str))

cv.get_feature_names()

Y = VietNamelist.Gender.values.astype(str)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

def predictGenderViet():
    svm = SVC()
    svm.fit(X_train, Y_train)
    pred=svm.predict(X_test)
    return pred

def makePrediction(name):

    # load the model
    loaded_model = pickle.load(open('SVM_model.pkl', 'rb'))

    # make a prediction
    result=(loaded_model.predict(cv.transform([name]))).astype(str)
    return result

