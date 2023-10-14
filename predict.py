import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import requests
import grpc
import tensorflow as tf



#set the model url
url = 'http://localhost:8501/v1/models/Feed19model:predict'
#load the data
df=pd.read_excel("OracleTags3.xlsx",sheet_name='Test19')

#change types
def change_types(df):
    for col in df.columns:
        if df[col].dtypes == 'object':
            df[col] = df[col].apply(pd.to_numeric,errors ='coerce')
    return df

df=change_types(df)

df=df.fillna(df.median())

df=df.reset_index(drop=True)


#defining sequence functions build

X=df['Feed19'].values.reshape(-1,1)

scaler=MinMaxScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)


seq_size=60


def to_sequences(x, y, seq_size):
    x_values, y_values = [], []

    for i in range(len(x) - seq_size):
        x_slice = x[i:i+seq_size]
        y_slice = y[i+seq_size]
        x_values.append(x_slice)
        y_values.append(y_slice)

    return np.array(x_values), np.array(y_values)

X19_test,y19_test=to_sequences(X_scaled,X_scaled,seq_size)


# Make prediction function
def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    
    try:
        json_response = requests.post(url, data=data, headers=headers)
        json_response.raise_for_status()  # Check for HTTP error
        predictions = json.loads(json_response.text)['predictions']
        return predictions
    except requests.exceptions.RequestException as e:
        print(f"Error making prediction request: {e}")
        return None

# Make predictions
predictions19 = make_prediction(X19_test)

X_scaled = X_scaled.reshape(-1)

#get mae
mae19 = np.mean(np.abs(predictions19-X19_test),axis=1)
df_out = pd.DataFrame(df[seq_size:])
df_out['mae']=mae19
df_out['max_mae']= 0.1
df_out['anomaly']=df_out['mae']>df_out['max_mae']

X_out = df_out['anomaly']

# Convert the numpy array to a Python list
python_list = X_out.tolist()

# Convert the Python list to a JSON-formatted string
json_string = json.dumps(python_list)

# Print the JSON-formatted string
print(json_string)



