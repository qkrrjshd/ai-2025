import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("wineqrev (1).csv")
X = df.iloc[:, 0:11]
y = df.iloc[:, 11]

mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(11,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
modelpath = 'wine_modelr.keras'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32,
                    callbacks=[early_stopping_callback, checkpointer])

score = model.evaluate(X_test, y_test)
print("Test MSE:", score[0])

pred = model.predict(X_test)
y_np = y_test.to_numpy()
print("\n[예측 vs 실제 품질값]")
for i in range(len(y_np)):
    print(f"예측: {pred[i][0]:.2f}, 실제: {y_np[i]}")
