# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Contributor/Developer**: Anirudh Kumar Rao

---

## Overview

**Content Based Recommendation System** - Given a Textual User Query the code will return Top_N (e.g 5) similar Movies present in the Dataset by computing the **Cosine Similarity Score** of the User Query and Text Features(e.g. Genre, Overview/Description, Director).

Given the dataset and query and depending of the model chosen (tfidf, bert or both) the Code will:
 - Perform Config lookups to set up runtime variables like:
   - Load the Dataset
   - Sample (N_Samples) rows from the dataset
   - Text_Feature_Columns (Fetch the Text column names we want to use in our analysis)
   - Top_N (set the number of recommendations we want returned)
 - Clean and Preprocess the Text Features
 - Create the TFIDF matrix or Load BERT model and create BERT embeddings for the cleaned text
 - Compute Similarity Scores and return the Dataframe with Top_N Similarity scores

## Requirements and Setup

1. Python Version : 3.11
   
2. Ensure you are in the root directory of this folder structure
    - lumaa-spring-2025-ai-ml (Root)
      - Code (Code Directory)
      - Data (Data Directory)

4. Libraries : Install the necessary Packages and Libraries from the requirements.txt file while in the root directory to set up your virtual environment
   ```shell
   pip install -r requirements.txt
   ```
5. Datasets:
   - The datasets used for this application are saved in the Data Directory of this Repository. Additionaly, they can also be obtained from Kaggle at the following links:
     - imdb_top_1000.csv : https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
     - imdb-movies-dataset.csv : https://www.kaggle.com/datasets/amanbarthwal/imdb-movies-data 
   
7. Update Config File (config.conf) in Code Directory as needed:
   All field values can be changed except for Data_Directory (should remain as Data)
    - Filename = CSV File we want to use as our Dataset (e.g imdb_top_1000.csv)
    - N_Samples = Randomly Samples N rows from the data (e.g. 500)
    - Text_Feature_Columns = Text Features to be used for computing TFIDF vectors and BERT embeddings (e.g. Genre, Overview, Director, Star1, Star2, Star3, Star4, Series_Title) (ensure atleast one value is present and last value should not end with a comma)
    - Movie_Title = Column Name of the Movie Title column in our selected dataset (e.g. Series_Title)
    - Top_N = Number of Recommendations we want returned (e.g. 5)
    - Random_Seed = Random Seed value to ensure replication of results across multiple runs (needed for N_Samples) (e.g. 5556)

> **Note**: Ensure you follow the same format for the values in the config file. If changing the File/Dataset we want to use, enusre appropriate changes are made to the Text_Feature_columns and Movie_Title fields (specific to each file)
---

## Running the Scripts
Once the Requirements and Setup has been completed, we can execute the code to generate the recommendations. Ensure you are in the Root directory and run the code using one of the below command:

1. Generate Recommendations using TFIDF
   ```shell
   python .\Code\Fetch_Recommendations.py -q "Some query you want to input" -m tfidf
   ```
   
2. Generate Recommendations using BERT
   ```shell
   python .\Code\Fetch_Recommendations.py -q "Some query you want to input" -m bert
   ```
   
3. Generate Recommendations using Both - Returns two dataframes with recommendations from each model
   ```shell
   python .\Code\Fetch_Recommendations.py -q "Some query you want to input" -m both
   ```
   
> **Note**: These commands will work only when we are in the Root Directory (lumaa-spring-2025-ai-ml or equivalent on your system). The --query (-q) user input is optional and has a default value of `"I like action movies set in space"`. The --model (-m) argument is **required and needs to be explicitly specified (tfidf, bert or both)**.

---

## Deliverables

1. **Short Video Demo**
   - Link : https://drive.google.com/file/d/1_0iUTjDkquqta2rzQ6GzjUaeq5Bi5sCs/view?usp=sharing
     
2. **Implementation of Solution**
   - Load and preprocess your dataset - Code/Load_and_Preprocess.py - load_base_dataset(), preprocess_text()  
   - Convert text data to vectors - Code/Load_and_Preprocess.py - fetch_vectorizer_and_tfidf()
   - Implement a function to compute similarity between the user’s query and each item’s description - Code/Fetch_Recommendations.py - calculate_cosine_similarity()
   - Return the top matches - Code/Fetch_Recommendations.py - recommend_movies_tfidf(), recommend_movies_bert()

3. **Results of Example**
   
     ```
      Movie Recommendations from TF-IDF Model :
                      Series_Title  Similarity_Score                         Genre                                           Overview  ...              Star2                Star3           Star4               Series_Title
    106                     Aliens          0.199292     Action, Adventure, Sci-Fi  Fifty-seven years after surviving an apocalypt...  ...      Michael Biehn          Carrie Henn     Paul Reiser                     Aliens
    183           Some Like It Hot          0.169884        Comedy, Music, Romance  After two male musicians witness a mob hit, th...  ...        Tony Curtis          Jack Lemmon     George Raft           Some Like It Hot
    114      2001: A Space Odyssey          0.153530             Adventure, Sci-Fi  After discovering a mysterious artifact buried...  ...      Gary Lockwood    William Sylvester  Daniel Richter      2001: A Space Odyssey
    692  The Man Who Would Be King          0.140892       Adventure, History, War  Two British former soldiers decide to set them...  ...      Michael Caine  Christopher Plummer   Saeed Jaffrey  The Man Who Would Be King
    389             The Iron Giant          0.124314  Animation, Action, Adventure  A young boy befriends a giant robot from outer...  ...  Harry Connick Jr.     Jennifer Aniston      Vin Diesel             The Iron Giant

     ```

     ```
      Movie Recommendations from BERT Model :
                   Series_Title  Similarity_Score                      Genre                                           Overview  ...             Star2           Star3               Star4             Series_Title
   745                  Gravity          0.458968    Drama, Sci-Fi, Thriller  Two astronauts work together to survive after ...  ...    George Clooney       Ed Harris    Orto Ignatiussen                  Gravity
   488               District 9          0.436793   Action, Sci-Fi, Thriller  Violence ensues after an extraterrestrial race...  ...       David James      Jason Cope      Nathalie Boltt               District 9
   339  Guardians of the Galaxy          0.429859  Action, Adventure, Comedy  A group of intergalactic criminals must pull t...  ...        Vin Diesel  Bradley Cooper         Zoe Saldana  Guardians of the Galaxy
   275             Blade Runner          0.428483   Action, Sci-Fi, Thriller  A blade runner must pursue and terminate four ...  ...      Rutger Hauer      Sean Young  Edward James Olmos             Blade Runner
   329              The Martian          0.427870   Adventure, Drama, Sci-Fi  An astronaut becomes stranded on Mars after hi...  ...  Jessica Chastain    Kristen Wiig           Kate Mara              The Martian

     ```

     
4. **Salary Expectation**
   - $25-$35/hr @ 20hrs/week ($2000-$2800/month)
   - Available to work 40hrs/week from May 10th - Aug 15th
  
 ---

## Future Scope/ Further Improvements

1. Use a composite Score of the TFIDF and BERT models for final recommendation.
2. Collaborative Filtering based on similiar users watch history (need to store user meta data and watch history)
