#%%
import configparser
import os
import re

import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_base_dataset() -> pd.DataFrame:
    """Load the movie dataset from the specified directory and file.

    This function reads a configuration file (`config.conf`) to retrieve the data directory and filename.
    It then constructs the file path and loads the dataset into a Pandas DataFrame.

    Returns:
    --------
    pd.DataFrame
        A Pandas DataFrame containing the movie dataset.

    """
    config = configparser.ConfigParser()
    config.read('Code\\config.conf')

    DATA_DIR = config.get('Data', 'Data_Directory')
    FileName = config.get('Data', 'Filename')
    n_samples = int(config.get('Data', 'N_Samples'))
    random_seed = int(config.get('Data', 'Random_Seed'))

    file_path = os.path.join(DATA_DIR, FileName)
    dataframe = pd.read_csv(file_path)
    dataframe = dataframe.sample(n=min(n_samples, dataframe.shape[0]), random_state=random_seed)

    return dataframe

def preprocess_text(text_data: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Clean and preprocess text by removing special characters, stopwords, and applying lemmatization.

    This function processes the input text by:
    - Converting it to lowercase.
    - Removing special characters and extra spaces.
    - Tokenizing the text and removing stopwords.
    - Applying lemmatization to normalize words.

    Parameters:
    -----------
    text_data : str
        The raw text data to be cleaned.
    lemmatizer : WordNetLemmatizer
        An instance of the NLTK WordNetLemmatizer used for lemmatization.
    stop_words : set
        A set of stopwords to remove from the text.

    Returns:
    --------
    str
        The cleaned and preprocessed text.

    """
    text = text_data.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split() # Split into Word tokens
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def fetch_vectorizer_and_tfidf(dataframe: pd.DataFrame) -> tuple:
    """
    Initialize a TF-IDF vectorizer and transform the movie descriptions into numerical vectors.

    This function fits a TF-IDF vectorizer on the 'Cleaned_Description' column of the dataset
    and returns the trained vectorizer along with the computed TF-IDF matrix.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        A Pandas DataFrame containing a column 'Cleaned_Description' with preprocessed text.

    Returns:
    --------
    tuple
        A tuple containing:
        - vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        - tfidf_matrix (sparse matrix): The transformed TF-IDF feature matrix.

    """
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Transform the movie descriptions into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(dataframe['Cleaned_Text'])
    return  vectorizer, tfidf_matrix
