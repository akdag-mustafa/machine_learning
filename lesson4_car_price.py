import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/mustafaakdag/Downloads/Audi_A1_listings.csv')
df.drop(columns=['index','href','MileageRank','PriceRank','PPYRank','Score'],inplace=True)
#preproccessing işlemleri
df.columns=["yil","kasa","mil","motor","ps","vites","yakit","sahip","fiyat","ppy"]
data=df
#print("ilk girilen :\n", data.head(3))
#en kolay değişecek olan araç motorunu numerik ifadeye çevirmek
data["motor"] = (
    df["motor"]
    .astype("string")                    # Metin tipine çevirir
    .str.replace("L", "", regex=False)   # L harfini kaldırır
    .str.strip()                         # Baştaki ve sondaki boşlukları kaldırır
)

data["motor"] = pd.to_numeric(
    data["motor"],
    errors="coerce"
)

data = pd.get_dummies(      #herşeyi numerik hale getirdik ml kullanımına uygun hale geldi
    data,
    columns=["kasa", "vites", "yakit"],
    drop_first=True,
    dtype=int,
)
print("son hali:\n",data.info())
y=data["fiyat"]
x=data.drop("fiyat",axis=1)
lm = LinearRegression()
model=lm.fit(x,y)

print(model.predict([[2017,30000,1.6,110,1,2600,0,1]]))
print(model.score(x,y))




