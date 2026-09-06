import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_sequences(values, sequence_length=5):
    X=[]; y=[]
    for i in range(len(values)-sequence_length):
        X.append(values[i:i+sequence_length]); y.append(values[i+sequence_length])
    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.float32)

def load_and_prepare_data(path,sequence_length=5,test_size=.2):
    df=pd.read_csv(path)
    if "traffic_flow" not in df.columns: raise ValueError("traffic.csv needs traffic_flow column")
    values=df[["traffic_flow"]].astype("float32").values
    split=int(len(values)*(1-test_size))
    scaler=MinMaxScaler(); scaler.fit(values[:split])
    scaled=scaler.transform(values)
    X,y=create_sequences(scaled,sequence_length)
    target_idx=np.arange(sequence_length,len(values))
    train=target_idx<split
    return X[train],X[~train],y[train],y[~train],scaler
