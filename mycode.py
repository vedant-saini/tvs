import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
df = pd.read_csv("training dataset.csv")
df
df.drop("Loan_ID", axis=1, inplace=True)
df.isna().sum()
na_vals = {}
print("----------NAN values can be replaced with----------")
for i in df.columns:
    if(pd.api.types.is_numeric_dtype(df[i])):
        na_vals[i]=df[i].median()
        plt.hist(df[i])
        plt.show()
    else:
        na_vals[i]=df[i].mode()[0]
    print(i," --> ",na_vals[i])

for i in df:
    df[i].fillna(na_vals[i], inplace=True)

# using median as the graphs show a skewed representation
print(df.isna().sum())
df
lbl_encoders = {}
for i in df:
    if(not(pd.api.types.is_numeric_dtype(df[i]))):
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i])
        lbl_encoders[i]=le
for column, encoder in lbl_encoders.items():
    print(column, " --> ", encoder.classes_)
df
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2002)
line = Pipeline([("Scalar",StandardScaler()), ("logistics",LogisticRegression())])
parameters = {"logistics__C":[0.00001,0.0001,0.001,0.01,0.1,1,10],"logistics__penalty":[None,'l2'],"logistics__solver":["newton-cg","newton-cholesky","lbfgs","saga"],"logistics__class_weight":[None,'balanced']}
gs = GridSearchCV(estimator=line,param_grid=parameters,cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
gs.fit(x_train,y_train)
print("parameters -->  ",gs.best_params_,"\n\n","train score-->  ", gs.best_score_)
print("test accuracy-->  ", gs.score(x_test,y_test))
model = Pipeline([("scalar", StandardScaler()),("logistic_reg",LogisticRegression(C=0.01, class_weight=None, penalty='l2', solver='newton-cg'))])
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
joblib.dump(model, 'model.pkl')
joblib.dump(lbl_encoders, 'lbl_encoders.pkl')
joblib.dump(na_vals, 'na_vals.pkl')

print("Model and related objects saved successfully.")
df2 = pd.read_csv("Test Dataset.csv")
df2
loan_id = df2.iloc[:,0]
df2=df2.iloc[:,1:]
for i in df2:
    df2[i].fillna(na_vals[i],inplace=True)
print(df2.isna().sum())
df2
for i in df2.columns:
    if(not(pd.api.types.is_numeric_dtype(df2[i]))):
        df2[i] = lbl_encoders[i].transform(df2[i])
df2
final_pred = model.predict(df2)
final_pred
df3 = pd.DataFrame({"Loan_Id": loan_id,"Loan_Status": final_pred})
df3
df3.to_csv("Predictions.csv",index=None)