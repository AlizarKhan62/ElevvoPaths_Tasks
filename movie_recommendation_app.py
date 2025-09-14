import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess movie and rating data"""
    try:
        rat = pd.read_csv('Movie Recommendation System/ratings.csv')
        mov = pd.read_csv('Movie Recommendation System/movies.csv')
        
        # Fix column name if needed
        if 'MovieId' in mov.columns:
            mov.rename(columns={'MovieId': 'movieId'}, inplace=True)
            
        return rat, mov
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, None

@st.cache_data
def create_user_item_matrix(ratings_df):
    """Create user-item matrix"""
    user_itm_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    user_itm_matrix.fillna(0, inplace=True)
    return user_itm_matrix

@st.cache_data
def compute_similarities(user_itm_matrix):
    """Compute user and item similarities"""
    usr_sim = cosine_similarity(user_itm_matrix)
    usr_sim_df = pd.DataFrame(usr_sim, index=user_itm_matrix.index, columns=user_itm_matrix.index)
    
    itm_sim = cosine_similarity(user_itm_matrix.T)
    itm_sim_df = pd.DataFrame(itm_sim, index=user_itm_matrix.columns, columns=user_itm_matrix.columns)
    
    return usr_sim_df, itm_sim_df

def recommend_movies_user_based(user_id, user_itm_matrix, usr_sim_df, movies_df, num_recommendations=5):
    """User-based collaborative filtering"""
    if user_id not in usr_sim_df.index:
        return pd.DataFrame(columns=['movieId', 'MovieTitle', 'Score'])
    
    sim_scores = usr_sim_df[user_id].drop(user_id)
    sim_users = sim_scores.sort_values(ascending=False).index

    weighted_ratings = pd.Series(dtype=float)
    for sim_user in sim_users[:10]:
        rat = user_itm_matrix.loc[sim_user]
        weighted_ratings = weighted_ratings.add(rat * sim_scores[sim_user], fill_value=0)

    user_rated = user_itm_matrix.loc[user_id]
    recommendations = weighted_ratings[user_rated == 0].sort_values(ascending=False)

    recommended_movies = recommendations.head(num_recommendations)
    result = movies_df[movies_df['movieId'].isin(recommended_movies.index)][['movieId', 'MovieTitle']].copy()
    result['Score'] = result['movieId'].map(recommended_movies)
    result = result.sort_values('Score', ascending=False)
    
    return result

def recommend_movies_item_based(user_id, user_itm_matrix, itm_sim_df, movies_df, num_recommendations=5):
    """Item-based collaborative filtering"""
    if user_id not in user_itm_matrix.index:
        return pd.DataFrame(columns=['movieId', 'MovieTitle', 'Score'])
    
    scores = pd.Series(itm_sim_df.dot(user_itm_matrix.loc[user_id]), index=user_itm_matrix.columns)
    
    # Drop movies the user already rated
    rated_mov = user_itm_matrix.loc[user_id][user_itm_matrix.loc[user_id] > 0].index
    scores = scores.drop(rated_mov, errors='ignore')

    recommended = scores.sort_values(ascending=False).head(num_recommendations)
    result = movies_df[movies_df['movieId'].isin(recommended.index)][['movieId', 'MovieTitle']].copy()
    result['Score'] = result['movieId'].map(recommended)
    result = result.sort_values('Score', ascending=False)
    
    return result

def get_user_ratings(user_id, ratings_df, movies_df):
    """Get movies rated by a specific user"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id].merge(
        movies_df[['movieId', 'MovieTitle']], on='movieId'
    ).sort_values('rating', ascending=False)
    return user_ratings

def main():
    # Title and description
    st.title("🎬 Movie Recommendation System")
    st.markdown("""
    Welcome to the Movie Recommendation System! This app uses collaborative filtering to suggest movies you might like.
    Choose between **User-Based** and **Item-Based** recommendation algorithms.
    """)

    # Load data
    with st.spinner("Loading movie data..."):
        ratings_df, movies_df = load_data()
    
    if ratings_df is None or movies_df is None:
        st.stop()

    # Create matrices and similarities
    with st.spinner("Computing similarities..."):
        user_itm_matrix = create_user_item_matrix(ratings_df)
        usr_sim_df, itm_sim_df = compute_similarities(user_itm_matrix)

    # Sidebar for user input
    st.sidebar.header("🎯 Recommendation Settings")
    
    # User selection
    user_ids = sorted(ratings_df['userId'].unique())
    selected_user = st.sidebar.selectbox(
        "Select User ID", 
        options=user_ids, 
        index=0,
        help="Choose a user ID to get personalized recommendations"
    )
    
    # Algorithm selection
    algorithm = st.sidebar.radio(
        "Recommendation Algorithm",
        options=["User-Based", "Item-Based"],
        help="""
        - **User-Based**: Recommends movies liked by similar users
        - **Item-Based**: Recommends movies similar to ones you've rated highly
        """
    )
    
    # Number of recommendations
    num_recs = st.sidebar.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=20,
        value=10,
        help="How many movie recommendations to show"
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"🍿 Recommendations for User {selected_user}")
        
        # Generate recommendations
        if algorithm == "User-Based":
            with st.spinner("Generating user-based recommendations..."):
                recommendations = recommend_movies_user_based(
                    selected_user, user_itm_matrix, usr_sim_df, movies_df, num_recs
                )
        else:
            with st.spinner("Generating item-based recommendations..."):
                recommendations = recommend_movies_item_based(
                    selected_user, user_itm_matrix, itm_sim_df, movies_df, num_recs
                )
        
        if not recommendations.empty:
            # Display recommendations with styling
            st.markdown(f"### Top {len(recommendations)} {algorithm} Recommendations")
            
            for idx, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <h4 style="margin: 0; color: #1f77b4;">🎬 {row['MovieTitle']}</h4>
                        <p style="margin: 5px 0; color: #666;">
                            Movie ID: {row['movieId']} | 
                            Recommendation Score: {row['Score']:.3f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show as DataFrame
            if st.checkbox("Show as table"):
                st.dataframe(
                    recommendations[['MovieTitle', 'movieId', 'Score']].round(3),
                    use_container_width=True
                )
        else:
            st.warning(f"No recommendations found for User {selected_user}")

    with col2:
        st.header(f"📊 User {selected_user} Profile")
        
        # Show user's rating history
        user_ratings = get_user_ratings(selected_user, ratings_df, movies_df)
        
        if not user_ratings.empty:
            st.markdown(f"**Total Movies Rated:** {len(user_ratings)}")
            st.markdown(f"**Average Rating:** {user_ratings['rating'].mean():.2f}")
            
            # Rating distribution
            st.markdown("**Rating Distribution:**")
            rating_counts = user_ratings['rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)
            
            # Recent ratings
            st.markdown("**Recent Ratings:**")
            recent_ratings = user_ratings.head(10)
            for _, movie in recent_ratings.iterrows():
                st.markdown(f"⭐ **{movie['rating']}/5** - {movie['MovieTitle']}")
                
        else:
            st.warning(f"No rating history found for User {selected_user}")

    # Dataset information
    with st.expander("📈 Dataset Information"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Total Movies", len(movies_df))
            
        with col_b:
            st.metric("Total Users", len(ratings_df['userId'].unique()))
            
        with col_c:
            st.metric("Total Ratings", len(ratings_df))
        
        st.markdown("**Sample Movies:**")
        st.dataframe(movies_df.head(), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🎬 Movie Recommendation System | Built with Streamlit & Scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
