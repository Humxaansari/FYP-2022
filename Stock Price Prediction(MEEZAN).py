#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as dr
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# In[9]:


dataset_train = pd.read_csv('D:\MeezanBankTrain.csv')
training_set = dataset_train.iloc[:, 4:5].values


# In[54]:


stock_price_train = dataset_train.iloc[:-120, 4:5].values


# In[55]:


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set).reshape(-1,1)


# In[56]:


X_train = []
y_train = []
for i in range(120, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[57]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[58]:


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, verbose='auto')


# In[18]:


history = regressor.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=10, verbose=0)


# In[19]:


# list all data in history
print(history.history.keys())


# In[20]:


# import matplotlib as plt
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# In[21]:


# summarize history for loss
plt.figure(figsize=(20, 6),dpi=80)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'],loc='upper right')
plt.grid()
plt.show()


# In[22]:


dataset_test = pd.read_csv('D:/MeezanBankTest.csv')


# In[23]:


dataset_test.replace(' ','',inplace=True)
dataset_test = dataset_test.dropna()


# In[24]:


data2 = dataset_test.iloc[:, 4:5].values
total_test = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)


# In[59]:


inputs = total_test[len(total_test) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


# In[60]:


X_test = []
for i in range(120, len(inputs)):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[61]:


predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)
predicted_stock_price_train.shape


# In[66]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 6),dpi=80)
plt.plot(dataset_test['Close'], color = 'red', label = 'Real Meezan Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Meezan Stock Price')
plt.title('Meezan Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Meezan Stock Price')
plt.legend()
plt.grid()
plt.show()


# In[63]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
testScore =(mean_squared_error(dataset_test['Close'], predicted_stock_price))
print('Test Score: %.2f RMSE' % (testScore))
print('Test Score: %.2f MAE' % mean_absolute_error(dataset_test['Close'], predicted_stock_price))
print('Test Score: %.2f R2' % r2_score(dataset_test['Close'],predicted_stock_price))


# In[64]:


print(mean_squared_error(stock_price_train, predicted_stock_price_train))
print(r2_score(stock_price_train, predicted_stock_price_train))


# In[69]:


plt.figure(figsize=(20, 6),dpi=80)
plt.plot(stock_price_train, color = 'red', label = 'Real Meezan Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Meezan Stock Price')
plt.title('Meezan Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Meezan Stock Price')
plt.legend()
plt.grid()
plt.show()


# In[70]:


real_data = [inputs[len(inputs) - 120:len(inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data , (real_data.shape[0], real_data.shape[1], 1))


# In[71]:


prediction = regressor.predict(real_data)
prediction = sc.inverse_transform(prediction)
print(f"Prediction : {prediction}")


# In[72]:


regressor_30 = Sequential()
regressor_30.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor_30.add(Dropout(0.2))
regressor_30.add(LSTM(units = 50, return_sequences = True))
regressor_30.add(Dropout(0.2))
regressor_30.add(LSTM(units = 50, return_sequences = True))
regressor_30.add(Dropout(0.2))
regressor_30.add(LSTM(units = 50))
regressor_30.add(Dropout(0.2))
regressor_30.add(Dense(units = 30))
regressor_30.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor_30.fit(X_train, y_train, epochs = 100, batch_size = 32, verbose='auto')


# In[73]:


real_data_30 = [inputs[len(inputs) - 120:len(inputs + 1), 0]]
real_data_30 = np.array(real_data_30)
real_data_30 = np.reshape(real_data_30 , (real_data_30.shape[0], real_data_30.shape[1], 1))


# In[74]:


prediction_30 = regressor_30.predict(real_data_30)
prediction_30 = sc.inverse_transform(prediction_30)
print(f"Prediction : {prediction_30.transpose()}")


# In[75]:


DataFrame = pd.DataFrame(prediction_30)
prediction  = DataFrame.T
Prediction_Total = pd.concat((dataset_test['Close'],prediction), axis=0,sort=True,ignore_index=True)


# In[76]:


plt.figure(figsize=(20, 6),dpi=80)
plt.plot(prediction, color = 'blue', label = 'Predicted Stock Price')
plt.title('Meezan Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Meezan Stock Price')
plt.legend()
plt.grid()
plt.show()


# In[77]:


plt.figure(figsize=(20, 6),dpi=80)
plt.plot(Prediction_Total, color = 'blue', label = 'Predicted Stock Price')
plt.plot(dataset_test['Close'], color = 'red',label='Real Stock Price')
plt.title('Meezan Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Meezan Stock Price')
plt.legend()
plt.grid()
plt.show()


# In[78]:


regressor_30.summary()


# In[40]:


import cufflinks as cf
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
import matplotlib.pyplot as plt


# In[41]:


cf.go_offline()
init_notebook_mode()


# In[42]:


TICKER = "Meezan"
dataset_train["Close"].plot(title=f"{TICKER}'s stock price",figsize=(20, 6),grid=True,xlabel='Time',ylabel='Price')


# In[43]:


qf = cf.QuantFig(dataset_train, title="Meezan's stock price in 2021", name='Meezan')
qf.iplot()


# In[44]:


fig = go.Figure(data=
    [go.Candlestick(x=dataset_train.index,
                    open=dataset_train["Open"],
                    high=dataset_train["High"],
                    low=dataset_train["Low"],
                    close=dataset_train["Close"])]
)

fig.update_layout(
    title=f"{TICKER}'s adjusted stock price",
    yaxis_title="Price ($)"
)

fig.show()


# In[45]:


qf = cf.QuantFig(dataset_train, title="Meezan's stock price in 2021", name='Meezan')
qf.add_sma(periods=14, column='Close', color='purple')
qf.iplot()


# In[46]:


qf = cf.QuantFig(dataset_train, title="Meezan's stock price in 2021", name='Meezan')
qf.add_sma([10, 50], width=2, color=['yellow', 'red'])
qf.iplot()


# In[47]:


qf.add_rsi(periods=14, color='green')
qf.iplot()


# In[48]:


qf.add_bollinger_bands(periods=20, boll_std=2 ,colors=['orange','grey'], fill=True)
qf.iplot()


# In[49]:


qf.add_volume()
qf.iplot()


# In[50]:


qf.add_macd()
qf.iplot()


# In[ ]:




