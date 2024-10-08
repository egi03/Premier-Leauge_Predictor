Model for predicting outcome of Premier Leauge match given teams and odds 
- made by Eugen Sedlar 


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tensorflow as tf
```

Load data


```python
data = pd.read_csv("filtered_data.csv")
```

Convert columns into numerical with LabelEncoder


```python
label_enc = LabelEncoder()
data['HomeTeam'] = label_enc.fit_transform(data['HomeTeam'])
data['AwayTeam'] = label_enc.fit_transform(data['AwayTeam'])
result_label_enc = LabelEncoder()
data['Result'] = result_label_enc.fit_transform(data['Result'])
```


```python
X = data[['HomeTeam', 'AwayTeam', '1', 'X', '2']]
y = data['Result']
```

Normalize odds (not necessary since odds are already normalized)


```python
scaler = StandardScaler()
X[['1', 'X', '2']] = scaler.fit_transform(X[['1', 'X', '2']])
```

Split data into training and testing data


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
```

Create and train model


```python
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=3, activation='linear')
    
])
```


```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```


```python
model.fit(X_train, y_train, epochs=50)
```

    Epoch 1/50
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 917us/step - accuracy: 0.4700 - loss: 1.4325
    Epoch 2/50
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 583us/step - accuracy: 0.3884 - loss: 1.1335
    .
    .
    .
    Epoch 50/50
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - accuracy: 0.5304 - loss: 0.9510
    




    <keras.src.callbacks.history.History at 0x28197551a00>



Test model


```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

    Test Accuracy: 58.96%
    


```python

```
