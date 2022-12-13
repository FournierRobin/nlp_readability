import streamlit as st
import pandas as pd
import numpy as np
import json

import keras
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


st.title("Predire la lisibilité d'un texte")

input_text = st.text_area(label="Marquez votre phrase ici")

model = load_model('models/dropout_model/dropout_model.h5')
model.compile(loss='mse', optimizer=keras.optimizers.Adam(
    learning_rate=0.001), metrics=['mae', 'mape'])
with open('models/dropout_model/dropout_model_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


if st.button(label="Prédire"):
    inpu_text_to_array = np.array([input_text])
    test_seq_to_pred = pad_sequences(
        tokenizer.texts_to_sequences(inpu_text_to_array), maxlen=180)

    prediction = model.predict(test_seq_to_pred)
    st.metric(label="Indice de lisibilité", value=prediction[0][0])
