import numpy as np # linear algebra
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Mobile Price Prediction Datatset.csv")
df = df.drop(['Unnamed: 0','Brand me'] , axis=1)
df = df[df['Price'] < 250000]
df[df['Selfi_Cam'] > 30]
df.isna().sum()
df['Selfi_Cam'] = df['Selfi_Cam'].fillna(df['Selfi_Cam'].median())
df = df[df['Selfi_Cam'] < 30]
df[df['RAM'] > 16]
# filling the ROM NA fields with the mode of the ROM column
df['ROM'] = df['ROM'].fillna(df['ROM'].mode()[0])
df['ROM'].isna().sum()
# filling the NA mode values with mode
df['Ratings'] = df['Ratings'].fillna(df['Ratings'].mode()[0])
df['Ratings'].isna().sum()
# setting the random seed so each time the code is run the peformance  don't change
np.random.seed(52)

# assigning the feature (x) value
x = df.drop('Price', axis=1)
x.head()

# assigning the target (y) value
y = df['Price']
y.head()



x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2)


model2 = RandomForestRegressor()
model2.fit(x_train, y_train)

model2.score(x_train, y_train)
# predicting the price on the testing set
y2_pred = model2.predict(x_test)
from sklearn.metrics import root_mean_squared_error, r2_score , mean_absolute_error

rmse = root_mean_squared_error(y_test, y2_pred)
r2Score = r2_score(y_test, y2_pred)
mae = mean_absolute_error(y_test, y2_pred)
print(" Random Forest Regressor Peformance : ")
print("Mean Squared Error : ", rmse)
print("R2 score: ", r2Score)
print("Mean Absolute Error: ", mae)
print("training score", model2.score(x_train, y_train))
print("testing score", model2.score(x_test, y_test))

pickle.dump(model2,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))