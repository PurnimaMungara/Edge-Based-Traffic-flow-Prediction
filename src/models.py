import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,LSTM,Dense,Dropout

def build_teacher(sequence_length,n_features):
    m=Sequential([Input((sequence_length,n_features)),LSTM(128,return_sequences=True),Dropout(.2),LSTM(64),Dense(32,activation="relu"),Dense(1)],name="Teacher_LSTM")
    m.compile(optimizer="adam",loss="mse",metrics=["mae"]); return m

def build_student(sequence_length,n_features):
    return Sequential([Input((sequence_length,n_features)),LSTM(32),Dense(16,activation="relu"),Dense(1)],name="Student_LSTM")
