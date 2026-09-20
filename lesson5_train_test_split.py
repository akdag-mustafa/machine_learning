import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("/Users/mustafaakdag/Downloads/Audi_A1_listings.csv")
df=df[["Year","Type","Mileage(miles)","Engine","PS","Transmission","Fuel","Number_of_Owners","Price(£)"]]
df.columns=["yil ","kasa","mil","motor","ps","vites","yakit","sahip","fiyat"]
df["motor"]=df["motor"].str.replace("L","")
df["motor"]=pd.to_numeric(df["motor"])
df=pd.get_dummies(df,columns=["kasa","vites","yakit"],drop_first=True,dtype=int)
y=df["fiyat"]
x=df.drop("fiyat",axis=1)
#train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=6)




lm = LinearRegression()
model=lm.fit(x_train,y_train)

print(model.score(x_test,y_test))
print(model.predict([[2016,30000,1.0,90,5,0,1]]))
