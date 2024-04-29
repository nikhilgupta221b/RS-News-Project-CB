from flask import Flask, render_template, request
import recommendation_engine
import random

app = Flask(__name__)

# Load the dataset and perform initial preprocessing
data = recommendation_engine.load_data()

@app.route('/')
def index():
    # Get 5 random news articles
    random_articles = random.sample(data['Title'].tolist(), 5)
    return render_template('index.html', random_articles=random_articles)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_article = request.form['article_title']
    recommendations = recommendation_engine.get_recommendations(user_article, data=data)
    return render_template('results.html', article=user_article, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
