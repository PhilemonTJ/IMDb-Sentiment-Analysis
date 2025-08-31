# IMDb Movie Review Sentiment Analysis

This project is a Python-based sentiment classification pipeline for IMDb movie reviews, using the `nltk` corpus and popular machine learning libraries.

## Project Objective

The primary goal of this notebook is to demonstrate an end-to-end process for building a sentiment classifier. The pipeline includes:
* Corpus Creation
* Text Preprocessing
* TF-IDF Vectorization
* Model Training and Evaluation

## Libraries and Dependencies

The following Python libraries are required to run this project:
* **pandas:** For data handling and manipulation.
* **numpy:** For numerical operations.
* **scikit-learn:** For machine learning models and evaluation metrics.
* **nltk:** For natural language processing tasks like tokenization, stop-word removal, and lemmatization.
* **seaborn & matplotlib:** For data visualization.
* **wordcloud:** To generate word clouds for visual analysis.

## Key Steps

1.  **Corpus Creation:** The notebook uses the built-in `nltk` `movie_reviews` corpus. The data is structured into a pandas DataFrame with two columns: `review` and `label` (positive or negative).

2.  **Text Preprocessing:** A custom function `preprocess_text()` is defined to clean the text data. This involves:
    * Converting text to lowercase.
    * Removing punctuation and numbers.
    * Tokenizing the text.
    * Removing English stopwords.
    * Lemmatizing the tokens.

3.  **Feature Extraction:** The preprocessed text is converted into numerical features using `TfidfVectorizer` from `scikit-learn`.

4.  **Model Training:** Two classification models are trained and evaluated on the dataset:
    * Multinomial Naive Bayes
    * Logistic Regression

5.  **Evaluation:** The performance of the models is evaluated using a classification report and a confusion matrix.

## How to Run the Notebook

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PhilemonTJ/IMDb-Sentiment-Analysis.git
    cd IMDb-Sentiment-Analysis
    ```
2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(A `requirements.txt` file should be created from the list of dependencies mentioned in the notebook)*
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook WA_CIA3_Comp2.ipynb
    ```

Feel free to explore the notebook, modify the code, and try different models or preprocessing techniques.
