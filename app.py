import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import gdown
import os
import tensorflow as tf

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Google Drive file IDs for each file
config_file_id = 'https://drive.google.com/file/d/1SjREj2zuzAS4ru4dLPo3wcp3PPShqsYT/view?usp=drive_link'  # Replace with your `config.json` file ID
generation_config_file_id = 'https://drive.google.com/file/d/1RmhbiMOmrXE3M0H8HrWZgfRODt3e-U8G/view?usp=drive_link'  # Replace with your `generation_config.json` file ID
model_weights_file_id = 'https://drive.google.com/file/d/1oaQepOGBLn993OxGJnPYy5Ga04h1ty8i/view?usp=drive_link'  # Replace with your `tf_model.h5` file ID

# Paths for downloaded files
model_path = "tf_model"
if not os.path.exists(model_path):
    os.makedirs(model_path)

config_file_path = os.path.join(model_path, "config.json")
generation_config_file_path = os.path.join(model_path, "generation_config.json")
model_weights_file_path = os.path.join(model_path, "tf_model.h5")

# Function to download files from Google Drive
def download_file(file_id, output_path):
    file_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    gdown.download(file_url, output_path, quiet=False)

# Download the files if they don't exist
if not os.path.exists(config_file_path):
    download_file(config_file_id, config_file_path)
if not os.path.exists(generation_config_file_path):
    download_file(generation_config_file_id, generation_config_file_path)
if not os.path.exists(model_weights_file_path):
    download_file(model_weights_file_id, model_weights_file_path)

# Load tokenizer from a pre-trained model (use the same as training)
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the model using the downloaded weights and configuration
model = TFAutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model_name_or_path=None,  # Specify None since we are loading custom weights
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
