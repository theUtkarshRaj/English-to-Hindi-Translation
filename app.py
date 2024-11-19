import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import gdown
import os
import tarfile
import tensorflow as tf

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Google Drive file ID and paths
file_id = '1VGMYPz_-EkAlE7RUBl_F64AOx_zZvLt5'  # Update this with your folder's file ID
model_path = 'tf_model'
model_file = os.path.join(model_path, "model.tar.gz")  # Adjust to match your model archive name

# Download and extract the model from Google Drive
if not os.path.exists(model_path):
    os.makedirs(model_path)
    model_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    st.info("Downloading the model...")
    gdown.download(model_url, model_file, quiet=False)
    
    # Extract the tar.gz file
    with tarfile.open(model_file, "r:gz") as tar:
        tar.extractall(path=model_path)
    st.success("Model downloaded and extracted!")

# Load the model and tokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"  # Pre-trained model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ensure proper path for the extracted model
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

# Streamlit app title and description
st.title("English to Hindi Translation")
st.markdown("This AI-powered translator converts English text to Hindi using a pre-trained model.")

# Text input widget
input_text = st.text_area("Enter English text")

# Translate button
if st.button("Translate"):
    if input_text:
        # Tokenize the input text
        tokenized = tokenizer([input_text], return_tensors='tf')

        # Generate translation
        out = model.generate(**tokenized, max_length=128)

        # Decode the translated output
        translated_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Display the translated text
        st.subheader("Translated Text")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
