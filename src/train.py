import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
#load dataset
data=pd.read_csv("data/student_score.csv")
X=data[["hours"]]
y=data["score"]
model=LinearRegression()
model.fit(X,y)
with open("model/model.pkl","wb") as f:
    pickle.dump(model,f)
print("model trained successfully")