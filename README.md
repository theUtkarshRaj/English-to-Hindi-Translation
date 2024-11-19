---

# English to Hindi Translation

This repository contains a machine learning application built using a pre-trained model for English-to-Hindi translation. The model uses the `Helsinki-NLP/opus-mt-en-hi` model for translation, with the added functionality of hosting the application using **Streamlit** for an interactive translation experience.

## Features

- **Pre-trained Model**: The application leverages the `Helsinki-NLP/opus-mt-en-hi` model for high-quality English to Hindi translation.
- **Streamlit Web App**: The app is built using Streamlit for an easy-to-use web interface. Users can input English text, and the app returns the translated Hindi text.
- **Easy Setup**: Simple setup instructions and requirements to get started quickly with the translation model.

## Installation

To run the project locally, follow these steps:

### Step 1: Clone the repository

```bash
git clone https://github.com/theUtkarshRaj/English-to-Hindi-Translation.git
cd English-to-Hindi-Translation
```

### Step 2: Install the required dependencies

Create a virtual environment and install the dependencies listed in `requirements.txt`:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate
# On MacOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Step 3: Run the application

After installing the dependencies, you can run the Streamlit app using the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser, and you can start translating English text to Hindi.

## Usage

Once the app is running, follow these steps to use it:

1. **Enter English Text**: Type or paste the text you want to translate in the input field.
2. **Click Translate**: After entering the text, click the "Translate" button.
3. **View Translated Text**: The translated text will appear below the input area, converted to Hindi.

## Model Details

The translation model used in this project is based on the `Helsinki-NLP/opus-mt-en-hi`, a pre-trained model for machine translation between English and Hindi. This model can handle both short and long text inputs.

## Technologies Used

- **Transformers**: The Hugging Face `transformers` library for loading and using the pre-trained translation model.
- **TensorFlow**: TensorFlow is used for loading and running the model.
- **Streamlit**: The Streamlit framework is used to create an interactive web interface for the translation service.
- **gdown**: Google Drive integration for downloading the pre-trained model files.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please follow the contribution guidelines and ensure your changes are properly documented.

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for the pre-trained `opus-mt-en-hi` model.
- [Streamlit](https://streamlit.io/) for providing an easy way to deploy machine learning apps.
- [Hugging Face](https://huggingface.co/) for the Transformers library.

---
