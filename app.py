import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import os

# Paths for manually uploaded files
config_file_path = "./config.json"
generation_config_file_path = "./generation_config.json"
model_weights_file_path = "./tf_model.h5"

# Load tokenizer from a pre-trained model (same as training)
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the model using the manually uploaded files
model = TFAutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model_name_or_path=None,  # Use None since we're loading custom weights
    config=config_file_path,
    weights=model_weights_file_path
)

# Streamlit app title and description
st.title("English to Hindi Translation")
st.markdown("This AI-powered translator converts English text to Hindi using a pre-trained model.")

# Text input widget
input_text = st.text_area("Enter English text")

# Translate button
if st.button("Translate"):
    if input_text:
        # Tokenize the input text
        tokenized = tokenizer([input_text], return_tensors="tf")

        # Generate translation
        out = model.generate(**tokenized, max_length=128)

        # Decode the translated output
        translated_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Display the translated text
        st.subheader("Translated Text")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
