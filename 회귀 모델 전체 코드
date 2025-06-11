## 회귀 모델 전체 코드
```python
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
df = pd.read_csv("wineqrev (1).csv")
X = df.iloc[:, 0:11]
y = df.iloc[:, 11]

# 정규화
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(11,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# 콜백 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
modelpath = 'wine_modelr.keras'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32,
                    callbacks=[early_stopping_callback, checkpointer])

# 평가
score = model.evaluate(X_test, y_test)
print("Test MSE:", score[0])

# 예측 결과 출력
pred = model.predict(X_test)
y_np = y_test.to_numpy()
print("\n[예측 vs 실제 품질값]")
for i in range(len(y_np)):
    print(f"예측: {pred[i][0]:.2f}, 실제: {y_np[i]}")
```
