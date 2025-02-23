#%%
import argparse
import configparser

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

from Load_and_Preprocess import fetch_vectorizer_and_tfidf, load_base_dataset, preprocess_text

nltk.download('stopwords')
nltk.download('wordnet')

def calculate_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors

    Formula:
    -------------------------
    cosine_similarity = dot(A.B) / (NORM(A)*NORM(B))

    Parameters:
    ----------
    vec1 : np.ndarray
        The first vector.
    vec2 : np.ndarray
        The second vector.

    Returns:
    -------
    float
        The cosine similarity score between vec1 and vec2. Ranges from -1 (opposite) to 1 (identical).
    """
    vecs_dot = np.dot(vector1, vector2)
    vec1_norm = np.linalg.norm(vector1)
    vec2_norm = np.linalg.norm(vector2)

    # Prevent Division by 0 error
    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0

    return vecs_dot/(vec1_norm*vec2_norm)


def recommend_movies_tfidf(user_input, df, vectorizer, tfidf_matrix, top_n=5) -> pd.DataFrame:
    """Recommend similar movies using TF-IDF vectorization.

    This function takes a user's input description, converts it into a TF-IDF vector,
    and finds the top N most similar movies from the dataset based on cosine similarity.

    Parameters:
    ----------
    user_input : str
        A textual description of the type of movie the user is interested in.
    df : pd.DataFrame
        A DataFrame containing movie details
    vectorizer : TfidfVectorizer
        A trained TF-IDF vectorizer used to transform text descriptions.
    tfidf_matrix : TF-IDF matrix
        The precomputed TF-IDF matrix of movie descriptions.
    top_n : int, optional
        The number of movie recommendations to return (default is 5).

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the top N recommended movies with their titles and overviews.

    """
    # Vectorize User Input and convert to Numpy Array
    user_vec = vectorizer.transform([user_input]).toarray().flatten()

    # Convert TF-IDF Matrix to np Array
    tfidf_matrix_np = tfidf_matrix.toarray()

    # Compute cosine similarity for the query against the Text descriptors for every movie
    similarity_scores = np.array([calculate_cosine_similarity(user_vec, movie_vec) for movie_vec in tfidf_matrix_np])

    # Get indices of top N most similar movies
    top_n_indices = similarity_scores.argsort()[::-1][:top_n]

    # Retrieve recommended movies and similarity scores
    recommendations = df.iloc[top_n_indices].copy()
    recommendations['Similarity_Score'] = similarity_scores[top_n_indices]

    return recommendations


def recommend_movies_bert(user_input, df, model, top_n=5) -> pd.DataFrame:

    """Recommend similar movies based on BERT embeddings.

    This function takes a user's input description, processes it (converts to lower case), converts it into a BERT embedding,
    and finds the top N most similar movies from the dataset based on cosine similarity.

    Parameters:
    ----------
    user_input : str
        A textual description of the type of movie the user is interested in.
    df : pd.DataFrame
        A DataFrame containing movie details and the precomputed BERT embeddings which are stored in the '_Embeddings' column.
    model : SentenceTransformer
        A pre-trained SentenceTransformer model for generating embeddings.
    top_n : int, optional
        The number of movie recommendations to return (default is 5).

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the top N recommended movies along with their similarity scores.

    """

    # Preprocess User Input - Only convert to lower
    query_cleaned = user_input.lower()

    # Generate BERT embedding for User Query and convert to Numpy array
    user_embedding = model.encode(query_cleaned, convert_to_tensor=True).cpu().numpy()

    # Convert stored embeddings to Numpy Array
    movie_embeddings = np.array([embedding.cpu().numpy() for embedding in df['_Embeddings'].values])

    # Compute cosine similarity for the query against the Text descriptors for every movie
    similarity_scores = np.array([calculate_cosine_similarity(user_embedding, movie_emb) for movie_emb in movie_embeddings])

    # Get indices of top N most similar movies
    top_n_indices = similarity_scores.argsort()[::-1][:top_n]

    # Retrieve recommended movies and similarity scores
    recommendations = df.iloc[top_n_indices].copy()
    recommendations['Similarity_Score'] = similarity_scores[top_n_indices]

    return recommendations

if __name__ == "__main__":
    # Fetch/Parse command-line arguments
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("-q","--query", type=str, default = "I like Action movies set in space", help="Enter a query for recommendations")
    parser.add_argument("-m", "--model", type=str, choices=["tfidf", "bert", "both"], required=True, help="Choose the model for recommendations: 'tfidf' or 'bert'")
    args = parser.parse_args()
    query = args.query  # Loads the query supplied by the user or will take the default value if -q/--query not supplied
    model = args.model # Indicator for which model to use (tfidf, bert, both)

    movies_dataframe = load_base_dataset()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    config = configparser.ConfigParser()
    config.read('Code//config.conf')

    Top_N = int(config.get('Data', 'Top_N'))
    Title_Column = config.get('Data', 'Movie_Title')
    Selected_Features = config.get('Data', 'Text_Feature_Columns').split(',')
    Selected_Features = [col.strip() for col in Selected_Features]

    # Clean and Preprocess Text data within each selected feature from config
    for column in Selected_Features:
        column_tag = "Cleaned_" + column
        movies_dataframe[column_tag] = movies_dataframe[column].apply(
            lambda x: preprocess_text(x, lemmatizer, stop_words))

    # Merge all cleaned text features into one column
    movies_dataframe["Cleaned_Text"] = movies_dataframe[[f"Cleaned_{col}" for col in Selected_Features]].agg(" ".join, axis=1)

    selected_columns = [Title_Column] + ["Similarity_Score"] + Selected_Features

    if model == "tfidf":
        vectorizer, tfidf_matrix = fetch_vectorizer_and_tfidf(movies_dataframe)
        recommended_movies = recommend_movies_tfidf(query, movies_dataframe, vectorizer, tfidf_matrix, top_n=Top_N)
        print("\n Movie Recommendations: ")
        print(recommended_movies[selected_columns])

    elif model == "bert":
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for each movie description
        movies_dataframe['_Embeddings'] = movies_dataframe['Cleaned_Text'].apply(
            lambda x: bert_model.encode(x, convert_to_tensor=True))

        recommended_movies_bert = recommend_movies_bert(query, movies_dataframe, bert_model, top_n=Top_N)
        print("\n Movie Recommendations: ")
        print(recommended_movies_bert[selected_columns])

    else:
        vectorizer, tfidf_matrix = fetch_vectorizer_and_tfidf(movies_dataframe)
        recommended_movies = recommend_movies_tfidf(query, movies_dataframe, vectorizer, tfidf_matrix, top_n=Top_N)
        print("\n Movie Recommendations from TF-IDF Model : ")
        print(recommended_movies[selected_columns])

        bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        movies_dataframe['_Embeddings'] = movies_dataframe['Cleaned_Text'].apply(
            lambda x: bert_model.encode(x, convert_to_tensor=True))

        recommended_movies_bert = recommend_movies_bert(query, movies_dataframe, bert_model, top_n=Top_N)
        print("\n Movie Recommendations from BERT Model : ")
        print(recommended_movies_bert[selected_columns])