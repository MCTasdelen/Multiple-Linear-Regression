import pandas as pd
import numpy as np
df=pd.read_csv("tenis_verisi.txt")
print(df)
#humidity i tahmin edicez
# Veri önişleme
# kategorik verileri numerice çevirme
outlook=df.iloc[:,0:1].values
print(outlook)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
outlook[:,0]=le.fit_transform(df.iloc[:,0])
print(outlook)
ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)
windy=df.iloc[:,3:4].values
windy[:,0]=le.fit_transform(df.iloc[:,3])
print(windy)
windy=ohe.fit_transform(windy).toarray()
# 1. kolonda 0:false 1:true 0.kolonda tam tersi
play=df.iloc[:,-1:].values
play[:,-1]=le.fit_transform(df.iloc[:,-1])
print(play)
play=ohe.fit_transform(play).toarray()
#1. kolonda 0:no,1:yes
#play ve windy için aslında label encoding yapmak daha doğru


##DataFrame oluşturma
temp=df.iloc[:,1:2].values
temp_df=pd.DataFrame(data=temp,
                     index=range(14),
                     columns=["temperature"])
outlook_df=pd.DataFrame(data=outlook
                        ,index=range(14)
                        ,columns=["overcast","rainy","sunny"] )
windy_df=pd.DataFrame(data=windy[:,1],
                      index=range(14),
                      columns=["windy"])
play_df=pd.DataFrame(data=play[:,1],
                     index=range(14),
                     columns=["play"])
concat_df=pd.concat([outlook_df,temp_df,windy_df,play_df],axis=1)

humidity=pd.DataFrame(data=df.iloc[:,2:3],
                      index=range(14),
                      columns=["humidity"])

all_df=pd.concat([outlook_df,temp_df,humidity,windy_df,play_df],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(concat_df,humidity,
                                                    test_size=0.33,
                                            random_state=0)
#Multi Linear Model 
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)

#Backward Elimination
import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int), values=concat_df,axis=1)
X_liste=concat_df.iloc[:,[0,1,2,3,4,5]].values

X_liste=np.array(X_liste,dtype=float)
model= sm.OLS(endog=humidity,exog=X_liste).fit()
print(model.summary())



X_liste=concat_df.iloc[:,[0,1,2,3,5]].values

X_liste=np.array(X_liste,dtype=float)
model= sm.OLS(endog=humidity,exog=X_liste).fit()
print(model.summary())

x_train.drop('windy',axis=1,inplace=True)
x_test.drop('windy',axis=1,inplace=True)

regressor.fit(x_train,y_train)
y_predict_result=regressor.predict(x_test)


