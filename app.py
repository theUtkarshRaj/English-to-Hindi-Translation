import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# Load model and tokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"  # Replace this if needed
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ensure the correct path to your saved model
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")  # Path to your model folder

# Streamlit app title and description
st.title("English to Hindi Translation")
st.markdown("This AI-powered translator converts English text to Hindi using a pre-trained model.")

# Text input widget
input_text = st.text_area("Enter English text")

# Translate button
if st.button("Translate"):
    if input_text:
        # Tokenize the input text
        tokenized = tokenizer([input_text], return_tensors='np')

        # Generate translation
        out = model.generate(**tokenized, max_length=128)

        # Decode the translated output
        translated_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Display the translated text
        st.subheader("Translated Text")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
