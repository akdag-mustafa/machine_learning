import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


data = pd.read_csv('/Users/mustafaakdag/Downloads/machine learning data/student_performance.csv')
data.rename(columns={"student_id":"ogrenci_no"},inplace=True)
yeniData=data.drop(columns=["ogrenci_no","gender","age"])
print(yeniData.head(3))
x=yeniData[["study_hours_per_week","attendance_rate"]]
y=yeniData[["final_score"]]
lm=LinearRegression()
model=lm.fit(x,y)
print("model katsayısı:" , model.coef_ )          #katsayı
print("model sabiti:" ,model.intercept_  )    #sabit
print(model.score(x,y))

