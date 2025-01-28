import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.h5')

with open('tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predict_word_index = np.argmax(predicted,axis = 1)[0]
    predicted_word = tokenizer.index_word.get(predict_word_index,None)

    return predicted_word


st.title("Next word prediction Using LSTM")
input_text = st.text_input("Enter your text")

if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word : {next_word}")