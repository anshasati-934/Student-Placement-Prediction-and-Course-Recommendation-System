from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

app = Flask(__name__)
CORS(app)

# Load the placement prediction model
with open('placement_model.pkl', 'rb') as f:
    placement_model = pickle.load(f)

# Load the salary prediction model
with open('salary_model.pkl', 'rb') as f:
    salary_model = pickle.load(f)


df = pd.read_csv('ultimate_course.csv')


def preprocess(text):
    text = text.lower()
    return text

df['Processed Description'] = df['Course Description'].apply(preprocess)


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Processed Description'])


placement_features = ['tier', 'cgpa', 'inter_gpa', 'ssc_gpa', 'internships', 'no_of_projects',
                      'is_participate_hackathon', 'is_participated_extracurricular',
                      'no_of_programming_languages', 'dsa', 'mobile_dev', 'web_dev',
                      'Machine Learning', 'cloud', 'CSE', 'ECE', 'MECH']


salary_features = ['tier', 'cgpa', 'internships', 'no_of_projects',
                   'is_participate_hackathon', 'is_participated_extracurricular',
                   'no_of_programming_languages', 'dsa', 'mobile_dev', 'web_dev',
                   'Machine Learning', 'cloud', 'CSE', 'ECE', 'MECH']


@app.route('/')
def index():
    
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        input_data = {feature: float(data.get(feature, 0)) for feature in placement_features}
        input_df = pd.DataFrame([input_data])

        # Predict placement probability
        placement_prob = placement_model.predict_proba(input_df)[0][1]

        
        salary_input_data = {feature: input_data[feature] for feature in salary_features}
        salary_input_df = pd.DataFrame([salary_input_data])

        # Predict salary
        salary_pred = salary_model.predict(salary_input_df)[0]

        return jsonify({
            'status': 'success',
            'placement_probability': round(placement_prob * 100, 2),  
            'predicted_salary': round(salary_pred, 2)  
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/recommend_courses', methods=['POST'])
def recommend_courses():
    try:
        data = request.json
        user_input = preprocess(data['input'])

        # Generate recommendation based on the input
        user_vector = vectorizer.transform([user_input])
        cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[-5:][::-1]

        recommended_courses = df.iloc[similar_indices][['Course Name', 'University', 'Difficulty Level', 'Course URL']].to_dict(orient='records')

        return jsonify(recommended_courses)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
NEWS_API_KEY = '7bdedc4358f14f23a14fe405738a36b1'

# Route to display news 
@app.route('/news')
def get_news():
    url = f'https://newsapi.org/v2/everything?q=placement+technology&apiKey={NEWS_API_KEY}'
    
    response = requests.get(url)
    news_data = response.json()

    if news_data['status'] == 'ok':
        articles = news_data['articles']
    else:
        articles = []

    return render_template('news.html', articles=articles)

data = pd.read_csv('naukri_com-job_sample.csv')


@app.route('/api/industry-data')
def industry_data():
    # Group data by industry
    industry_counts = data['industry'].value_counts().head(10).reset_index()
    industry_counts.columns = ['industry', 'count']

    
    return jsonify(industry_counts.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

