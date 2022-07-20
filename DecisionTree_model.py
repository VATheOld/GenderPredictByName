import pandas as pd
import numpy as np
import pickle

co_l = ["Name", "Gender"]
engNamelist = pd.read_csv('Eng_name_gender_dataset.csv', usecols=co_l)
engNamelist.Gender.replace({"M": 0, "F": 1}, inplace=True)

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],  # First letter
        'first2-letters': name[0:2],  # First 2 letters
        'first3-letters': name[0:3],  # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }


features = np.vectorize(features)

# Extract the features for the dataset
df_X = features(engNamelist['Name'])

df_y = engNamelist['Gender']

from sklearn.feature_extraction import DictVectorizer

corpus = features(["Mike", "Julia"])
dv = DictVectorizer()
dv.fit(corpus)
transformed = dv.transform(corpus)

dv.get_feature_names()

# Train Test Split
from sklearn.model_selection import train_test_split

dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

dv = DictVectorizer()
dv.fit_transform(dfX_train)

# Model building Using DecisionTree

from sklearn.tree import DecisionTreeClassifier
def predictGenderEng():
    dclf = DecisionTreeClassifier()
    my_xfeatures = dv.transform(dfX_train)
    dclf.fit(my_xfeatures, dfy_train)
    test_xfeatures = dv.transform(dfX_test)
    y_pred = dclf.predict(test_xfeatures)

def makePredictionEng(name):

    # load the model
    loaded_model = pickle.load(open('DecisionTree_Model.pkl', 'rb'))

    # make a prediction
    vectorName=dv.transform(features(name)).toarray()
    result = loaded_model.predict(vectorName)

    return result

