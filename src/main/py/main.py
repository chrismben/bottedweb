import model as m
import pickle


model = m.Model()

classifier_path = 'lib/model/DecisionTreeClassifier.pkl'
clean_data_path = 'lib/model/CleanData.pk1'

with open(classifier_path, 'rb') as f1:
    model.clf = pickle.load(f1)

with open(clean_data_path, 'rb') as f2:
    model.vectorizer = pickle.load(f2)

#get data, clean it, and then predict.
model.predict("cleaned_data")

