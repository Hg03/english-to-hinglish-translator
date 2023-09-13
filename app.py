import streamlit as st
import os
import sys
import keras
import transformers
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay, AutoModelForSeq2SeqLM

st.set_page_config(layout='centered',page_title='Hinglish Generator')

@st.cache_data
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("model",from_tf=True)
    return (tokenizer, model)

def generate_text(input_text, tokenizer, model):
    output_text = ''
    tokenized = tokenizer([input_text],return_tensors='np')
    out = model.generate(**tokenized, max_length=128)
    with tokenizer.as_target_tokenizer():
        output_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return output_text

def main():
    st.markdown("## :blue[English] to :green[Hinglish] Text Translator")
    input_text = st.text_input('Put your text in English Language',placeholder='How is the weather ?')
    if st.button("Generate Hinglish Text"):
        tokenizer, model = load_model()
        output_text = generate_text(input_text,tokenizer, model)
        st.write(output_text)    

main()
    
