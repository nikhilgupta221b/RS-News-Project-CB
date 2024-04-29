import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load the dataset and perform initial preprocessing"""
    data = pd.read_csv('../static/news.tsv', header=None, sep='\t')
    data.columns = ["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"]
    data.loc[data['Title'].isnull(), 'Title'] = ''
    data.loc[data['Abstract'].isnull(), 'Abstract'] = ''
    data['combined_text'] = data['Title'] + ' ' + data['Abstract']
    return data

def get_recommendations(title, data=None, num_recommendations=5):
    if data is None:
        data = load_data()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

    article_index = data[data['Title'].str.lower() == title.lower()].index.tolist()
    if not article_index:
        return [("No matching articles found.", 0.0)]

    similarities = cosine_similarity(tfidf_matrix[article_index], tfidf_matrix)
    top_indices = similarities.argsort()[0][-num_recommendations - 1:-1]
    similar_articles = [(data['Title'].iloc[i], float(similarities[0][i])) for i in top_indices if i != article_index[0]]
    return similar_articles
