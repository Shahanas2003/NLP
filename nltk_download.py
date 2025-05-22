

import nltk
import os

# Define a local download path
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")

# Add it to NLTK's path
nltk.data.path.append(NLTK_DATA_DIR)

# Download required models to the local path
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
nltk.download("punkt")
nltk.download("stopwords")

