import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data_sp = pd.read_csv('healthcare-dataset-stroke-data.csv')
data_sp = data_sp.dropna()
y = data_sp['stroke']

# cleaning data
X = data_sp.drop(columns=['id', 'stroke'])

lb = LabelEncoder()
categories_list = ["gender","ever_married","work_type","Residence_type","smoking_status"]

for category in categories_list:
    X[category] = lb.fit_transform(X[category].astype(str))

# pre-processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=32)

print(f"Size X_train: {len(X_train)}")
print(f"Size X_test: {len(X_test)}")

# pipeline
l_regression_pipeline = Pipeline(steps=[
  ("normalize", MinMaxScaler()),  
  ("l_regression", LogisticRegression(class_weight='balanced'))
])

l_regression_pipeline.fit(X_train, y_train)

y_pred = l_regression_pipeline.predict(X_test)
y_pred_prob = l_regression_pipeline.predict_proba(X_test)
print(f"Acurácia de treinamento: {l_regression_pipeline.score(X_train, y_train)}")

#report
classes_name = ['no stroke', 'stroke']
report = classification_report(y_test, y_pred, target_names=classes_name)
print("Relatório de classificação:")
print(report)

mat_conf = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:")
print(mat_conf)