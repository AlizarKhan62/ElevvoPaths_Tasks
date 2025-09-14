# 🎬 Movie Recommendation System - Streamlit App

A collaborative filtering-based movie recommendation system deployed with Streamlit.

## Features

- **User-Based Collaborative Filtering**: Recommends movies liked by similar users
- **Item-Based Collaborative Filtering**: Recommends movies similar to highly-rated ones
- **Interactive Web Interface**: Easy-to-use Streamlit dashboard
- **User Profiles**: View rating history and preferences
- **Real-time Recommendations**: Get personalized movie suggestions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run movie_recommendation_app.py
```

### 3. Open in Browser
The app will automatically open at `http://localhost:8501`

## Dataset

- **Movies**: `Movie Recommendation System/movies.csv`
- **Ratings**: `Movie Recommendation System/ratings.csv`

## How It Works

1. **Data Loading**: Loads movie and rating datasets
2. **Matrix Creation**: Creates user-item rating matrix
3. **Similarity Computation**: Calculates user and item similarities using cosine similarity
4. **Recommendation Generation**: Provides personalized recommendations based on selected algorithm

## Deployment Options

### Local Development
```bash
streamlit run movie_recommendation_app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository

### Heroku
```bash
# Create Procfile
echo "web: streamlit run movie_recommendation_app.py --server.port \$PORT" > Procfile

# Deploy to Heroku
git add .
git commit -m "Add Streamlit app"
heroku create your-app-name
git push heroku main
```

## Performance Metrics

- **Dataset Size**: 100,000+ ratings
- **Users**: 600+ users  
- **Movies**: 9,000+ movies
- **Algorithm**: Collaborative Filtering with Cosine Similarity

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Pandas, Scikit-learn
- **Algorithms**: User-Based CF, Item-Based CF
- **Deployment**: Streamlit Cloud, Heroku, AWS
