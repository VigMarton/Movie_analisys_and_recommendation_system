import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,QuantileTransformer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Load data
movies_path = r"C:\Users\Admin\Desktop\Movie\data\df_released.csv"
not_relased_movies_path= r"C:\Users\Admin\Desktop\Movie\data\df_not_released.csv"

df_movies = pd.read_csv(movies_path, low_memory=False)
df_future_movies=pd.read_csv(not_relased_movies_path, low_memory=False)
df_new_releases=df_movies.sort_values(by="popularity",ascending= False).set_index("title").head(8)
# Page Configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon=":movie_camera:",  # You can use emojis or path to an icon image
    layout="wide"
)

# Custom CSS for background image and alignment
st.markdown("""
<style>
.stApp {
    background-image: url('https://img.freepik.com/free-photo/view-3d-film-reel_23-2151069393.jpg?t=st=1722242560~exp=1722246160~hmac=c2eb129a4e7805e9bfd0505e1028c99edfb80f6c5d525296b80400860e0d26cf&w=740');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    position: relative;
}

.stApp::before {
    content: "";
    background-color: rgba(0, 0, 0, 0.5); /* Adjust the transparency here */
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1;
}

.stApp > div {
    position: relative;
    z-index: 2;


.title-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 50px;
    background-color: #F5F5F5;
    border-bottom: 5px solid #FFD700; /* Accent color */
    color: #ffffff; /* Text color */
    padding: 20px;
    border-radius: 15px;
    margin: 2px;
    z-index: 3; /* Ensure this is above the background overlay */
}

.title-container h1 {
    font-size: 3rem;
    margin: 0;
    font-family: 'Roboto', sans-serif;
}

@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
}

.label {
    font-size: 1rem;
    color: #FFD700; /* Gold color for dark background */
    margin-right: 10px;
    display: inline-block;
    vertical-align: middle;
}

/* Adjust the margin and display properties for the select boxes */
.css-1n76uvr {
    background-color: #F5F5F5 !important;
    color: #333333 !important;
    border: 1px solid #FFD700 !important;
    border-radius: 5px !important;
    margin-top: -7px; /* Adjust this value to align vertically */
    display: inline-block;
    vertical-align: middle;
}

.css-1n76uvr:hover {
    border-color: #ffcc00 !important;
}

.css-1n76uvr:focus {
    border-color: #ffcc00 !important;
    box-shadow: 0 0 0 0.2rem rgba(255, 204, 0, 0.25) !important;
}

.css-1n76uvr option {
    background-color: #ffffff !important;
    color: #333333 !important;
}

.css-1n76uvr:disabled {
    background-color: #f0f0f0 !important;
    color: #cccccc !important;
}

.css-1n76uvr::placeholder {
    color: #cccccc !important;
}

.css-1n76uvr .dropdown {
    background-color: #F5F5F5 !important;
    color: #333333 !important;
}

.css-1n76uvr .dropdown:hover {
    background-color: #ffcc00 !important;
    color: #ffffff !important;
}

.css-1n76uvr .dropdown:focus {
    background-color: #ffcc00 !important;
    color: #ffffff !important;
}

.stSlider .stSliderTrack {
    background-color: #ffcc00 !important;
}

.stSlider .stSliderTrackFill {
    background-color: #FFD700 !important;
}

.stSlider .stSliderThumb {
    background-color: #ffffff !important;
    border: 2px solid #FFD700 !important;
}

.stSlider .stSliderThumb:hover {
    border-color: #ffcc00 !important;
}
</style>
""", unsafe_allow_html=True)



# User interface for selecting a movie title, genres, and actors
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("<br><span class='label'> Select a movie (optional):</span>", unsafe_allow_html=True)
with col2:
    movie_options = np.insert(df_movies['title'].unique(), 0, 'None')
    selected_movie = st.selectbox("", options=movie_options)

col3, col4 = st.columns([1, 3])
with col3:
    st.markdown("<br><span class='label'> Select genres (optional):</span>", unsafe_allow_html=True)
with col4:
    genre_options = np.sort(df_movies['genres'].str.split(',', expand=True).stack().unique(), 0)
    selected_genres = st.multiselect("", options=genre_options)

col5, col6 = st.columns([1, 3])
with col5:
    st.markdown("<br><span class='label'> Select actor(s) (optional):</span>", unsafe_allow_html=True)
with col6:
    actor_options = np.sort(np.unique(df_movies['actors'].dropna().apply(lambda x: x.split(', ')[:1]).explode()))
    selected_actors = st.multiselect("", options=actor_options)

col7, col8 = st.columns([1, 3])
with col7:
    st.markdown("<br><span class='label'> Select the number of recommendations:</span>", unsafe_allow_html=True)
with col8:
    num_recommendations = st.slider("", 1, 20, 8)



# The main function to make recommendations
def create_dummy_columns(df, names_list, column_name, df_column):
    """Create dummy columns for categorical variables."""
    for name in names_list:
        if name:
            name = name.strip().lower()
            df[column_name + name] = df[df_column].apply(lambda x: 1 if isinstance(x, str) and name in x.lower() else 0)

def get_recommendations(movie_title, selected_genres, selected_actors, num_recs):
    df = df_movies.copy()
    df_unreleased = df_future_movies.copy()
    
    if movie_title != 'None':
        selected_movie_df = df[df['title'] == movie_title].iloc[0]
        index_movie = df[df['title'] == movie_title].index[0]

        # Add the selected movie to the unreleased dataframe
        df_unreleased = pd.concat([df_unreleased, selected_movie_df.to_frame().T], ignore_index=True)

        # Extract relevant information
        actors_in_chosen_film = set(selected_movie_df['actors'].split(", ")[:8])
        directors_in_chosen_film = selected_movie_df['directors'].split(", ")
        writers_in_chosen_film = selected_movie_df['writers'].split(", ")[:2]
        genres = selected_movie_df['genres'].split(",")
        prod_companies = selected_movie_df['production_companies'].split(", ")[:1]
        keywords_in_chosen_movie = selected_movie_df['keywords'].split(", ")
        producers_in_chosen_movie = selected_movie_df['producer'].split(", ")
        original_language = selected_movie_df['original_language'].split(", ")
        
        # Create dummy columns for both dataframes
        create_dummy_columns(df, actors_in_chosen_film, 'actor_', 'actors')
        create_dummy_columns(df, directors_in_chosen_film, 'director_', 'directors')
        create_dummy_columns(df, writers_in_chosen_film, 'writer_', 'writers')
        create_dummy_columns(df, genres, 'genre_', 'genres')
        create_dummy_columns(df, prod_companies, 'production_company_', 'production_companies')
        create_dummy_columns(df, keywords_in_chosen_movie, 'keyword_', 'keywords')
        create_dummy_columns(df, producers_in_chosen_movie, 'producer_', 'producer')
        create_dummy_columns(df, original_language, 'language_', 'original_language')

        create_dummy_columns(df_unreleased, actors_in_chosen_film, 'actor_', 'actors')
        create_dummy_columns(df_unreleased, directors_in_chosen_film, 'director_', 'directors')
        create_dummy_columns(df_unreleased, writers_in_chosen_film, 'writer_', 'writers')
        create_dummy_columns(df_unreleased, genres, 'genre_', 'genres')
        create_dummy_columns(df_unreleased, prod_companies, 'production_company_', 'production_companies')
        create_dummy_columns(df_unreleased, keywords_in_chosen_movie, 'keyword_', 'keywords')
        create_dummy_columns(df_unreleased, producers_in_chosen_movie, 'producer_', 'producer')
        create_dummy_columns(df_unreleased, original_language, 'language_', 'original_language')

        # Prepare the feature matrices
        X = df.drop(columns=['original_language','twosite_rating', "averageRating", "popularity", 'runtime', 'vote_average',])
        X = X.select_dtypes(include=np.number)
        X_unreleased = df_unreleased.drop(columns=['original_language' ,"averageRating", "popularity", 'runtime', 'vote_average',])
        X_unreleased = X_unreleased.select_dtypes(include=np.number)

        # Scale the feature matrices
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_unreleased_scaled = scaler.transform(X_unreleased)

        # Fit the NearestNeighbors model
        model = NearestNeighbors(n_neighbors=len(X), algorithm='auto', metric='cosine')
        model.fit(X_scaled)

        # Get distances and indices of recommendations
        distances, recommendation_indices = model.kneighbors(X=[X_scaled[index_movie]], return_distance=True)

        # Get recommendations for released movies
        recommendations = df.iloc[recommendation_indices[0]]
        recommendations['distance'] = distances[0]

        # Fit the NearestNeighbors model for unreleased movies
        model_unreleased = NearestNeighbors(n_neighbors=len(X_unreleased), algorithm='auto', metric='cosine')
        model_unreleased.fit(X_unreleased_scaled)

        # Get distances and indices of recommendations for unreleased movies
        distances_unreleased, recommendation_indices_unreleased = model_unreleased.kneighbors(X=[X_unreleased_scaled[-1]], return_distance=True)

        # Get recommendations for unreleased movies
        recommendations_unreleased = df_unreleased.iloc[recommendation_indices_unreleased[0]].copy()
        recommendations_unreleased['distance'] = distances_unreleased[0]

        
        # Adjust distances based on selected genres
        if selected_genres:
            recommendations['genre_matches'] = recommendations['genres'].apply(
                lambda x: sum(genre.strip() in [g.strip() for g in x.split(",")] for genre in selected_genres))
            recommendations['distance'] = recommendations.apply(
                lambda row: row['distance'] * (0.1 ** row['genre_matches']), axis=1)
            
            recommendations_unreleased['genre_matches'] = recommendations_unreleased['genres'].apply(
                lambda x: sum(genre.strip() in [g.strip() for g in x.split(",")] for genre in selected_genres))
            recommendations_unreleased['distance'] = recommendations_unreleased.apply(
                lambda row: row['distance'] * (0.1 ** row['genre_matches']), axis=1)
        
        # Adjust distances based on selected actors
        if selected_actors:
            recommendations['actor_matches'] = recommendations['actors'].dropna().apply(
                lambda x: sum(actor.strip() in [a.strip() for a in x.split(",")] for actor in selected_actors))
            recommendations['distance'] = recommendations.apply(
                lambda row: row['distance'] * (0.1 ** row['actor_matches']), axis=1)
            
            recommendations_unreleased['actor_matches'] = recommendations_unreleased['actors'].apply(
                lambda x: sum(actor.strip() in [a.strip() for a in x.split(",")] for actor in selected_actors))
            recommendations_unreleased['distance'] = recommendations_unreleased.apply(
                lambda row: row['distance'] * (0.1 ** row['actor_matches']), axis=1)
        
        # Sort recommendations by adjusted distance and set index to 'title'
        recommendations = recommendations.sort_values(by='distance').head(num_recs+1)
        recommendations_unreleased = recommendations_unreleased.sort_values(by='distance').head(5)
        
        recommendations = recommendations.set_index("title")
        recommendations_unreleased = recommendations_unreleased.set_index("title")
    else:
        recommendations = df.copy().set_index('title')
        recommendations_unreleased = df_future_movies.copy().set_index('title')

        recommendations['genre_matches'] =0
        recommendations_unreleased['genre_matches'] =0
        recommendations['actor_matches'] =0
        recommendations_unreleased['actor_matches'] =0

        if selected_genres:
            recommendations['genre_matches'] = recommendations['genres'].apply(
                lambda x: sum(genre.strip() in [g.strip() for g in x.split(",")] for genre in selected_genres))
            recommendations_unreleased['genre_matches'] = recommendations_unreleased['genres'].apply(
                lambda x: sum(genre.strip() in [g.strip() for g in x.split(",")] for genre in selected_genres))
        else:
            recommendations['genre_matches'] = 0
            recommendations_unreleased['genre_matches'] = 0

        # Calculate matches for selected actors
        if selected_actors:
            recommendations['actors']=recommendations['actors'].astype(str)
            recommendations['actor_matches'] = recommendations['actors'].apply(
                lambda x: sum(actor.strip() in [a.strip() for a in x.split(",")] for actor in selected_actors))
            recommendations_unreleased['actor_matches'] = recommendations_unreleased['actors'].apply(
                lambda x: sum(actor.strip() in [a.strip() for a in x.split(",")] for actor in selected_actors))
        else:
            recommendations['actor_matches'] = 0
            recommendations_unreleased['actor_matches'] = 0

        # Calculate total matches
        recommendations['total_matches'] = recommendations['actor_matches'] + recommendations['genre_matches']
        recommendations_unreleased['total_matches'] = recommendations_unreleased['actor_matches'] + recommendations_unreleased['genre_matches']

        # Sort recommendations
        recommendations = recommendations.sort_values(by=['total_matches', 'twosite_rating'], ascending=False)
        recommendations_unreleased = recommendations_unreleased.sort_values(by=['total_matches', 'popularity'], ascending=False)
        recommendations=recommendations.head(adjusted_num_recs)
        recommendations_unreleased=recommendations_unreleased.head(8)        


        
        
    return recommendations, recommendations_unreleased





st.markdown("""
<style>
.head_movie {
    padding: 20px;
    background-color: rgba(0, 0, 0, 0.7); /* Dark background for contrast */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 0 15px rgba(255, 255, 255, 0.9);
    border-radius: 10px; /* Rounded corners */
}
.head_movie h2 {
    color: #FFD700;
    font-size: 24px; /* Larger font for titles */
    margin-bottom: 10px; /* Spacing below title */
}
.head_movie li, .head_movie p span {
    font-size: 16px; /* Slightly larger font for details */
    color: #FFD700;
}
.head_movie li span {
    font-weight: bold;
}
.head_movie p {
    font-size: 16px;
    color: #FFD700;
}
.blur-container {
    background-color: rgba(255, 255, 255, 0.1); /* Light background with transparency */
    backdrop-filter: blur(10px); /* Blur effect */
    padding: 20px; /* Padding inside the container */
    border-radius: 10px; /* Rounded corners */
    box-shadow: 
            0 4px 8px rgba(111, 111, 111, 0.7),  /* Glow on the bottom */
            0 4px 8px rgba(111, 111, 111, 0.7),  /* Glow on the right */
            0 4px 8px rgba(111, 111, 111, 0.7); /* Glow on the left */
    margin-top: -80px; /* Margin at the top */
}

.blur-container h2 {

    font-size: 20px; /* Slightly smaller font for recommendations */
    text-align: center;
    margin-bottom: 10px; /* Spacing below title */
}
.blur-container p {
    font-size: 14px;
    color: #333; /* Dark text color for readability */
}
</style>
""", unsafe_allow_html=True)



# Inject the CSS into the Streamlit app
st.markdown("""
<style>
    .custom-separator {
        border-top: 2px solid #FFD700;
        margin: 40px 0;
    }
</style>
""", unsafe_allow_html=True)



if st.button("Get Recommendations"):
    with st.spinner('Finding the best recommendations for you...'):
        # Fetch recommendations, potentially with one extra to exclude the selected movie later
        adjusted_num_recs = num_recommendations
        
        recommendations, unreleased_recommendations= get_recommendations(selected_movie, selected_genres, selected_actors, adjusted_num_recs)

        if selected_movie != 'None':
            movie_details = df_movies[df_movies['title'] == selected_movie].iloc[0]
            first_actor = movie_details['actors'].split(',')[:2] if movie_details['actors'] else None
            first_director = movie_details['directors'].split(',')[0] if movie_details['directors'] else None 
            main_recommendations, unreleased_recommendations = get_recommendations(selected_movie, selected_genres, selected_actors, adjusted_num_recs)
            selected_movie_row = df_movies[df_movies['title'] == selected_movie]
            recommendations = recommendations[recommendations.index != selected_movie]
            recommendations = recommendations.head(adjusted_num_recs)
            unreleased_recommendations = unreleased_recommendations[1:]

            if not selected_movie_row.empty and selected_movie_row.iloc[0]['actors']:
                first_actor = selected_movie_row.iloc[0]['actors'].split(',')[0].strip()
            else:
                first_actor = None
            if first_actor:
                actor_recommendations = df_movies[df_movies['actors'].apply(lambda x: first_actor in x.split(',') if pd.notna(x) else False)].set_index('title').head()
                actor_based_recommendations=actor_recommendations[actor_recommendations.index != selected_movie].head(4)
        
            else:
                actor_recommendations = pd.DataFrame()

            if not selected_movie_row.empty and selected_movie_row.iloc[0]['directors']:
                first_director = selected_movie_row.iloc[0]['directors'].split(',')[0].strip()
            else:
                first_director = None
            if first_director:
                director_recommendations = df_movies[df_movies['directors'].apply(lambda x: first_director in x.split(',') if pd.notna(x) else False)].set_index('title').head()
                director_based_recommendations=director_recommendations[director_recommendations.index != selected_movie].head(4)
            else:
                director_recommendations = pd.DataFrame()
            # Add the custom separator
            st.markdown('</div><div class="custom-separator"></div></div>', unsafe_allow_html=True)
            with st.container():
                col1, col2 = st.columns([5, 7])

                with col1:
                    if movie_details['poster_path'] == 'Unknown':
                        image_path="https://images.unsplash.com/photo-1619518594466-5bfc2dcbb83d?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    else:
                        image_path = 'https://image.tmdb.org/t/p/original/' + movie_details['poster_path']
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 75%; border-radius:14px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom: px;'></div>", unsafe_allow_html=True)

                with col2:
                    title = str(movie_details['title'])
                    runtimeMinutes = str(movie_details['runtime'])
                    startYear = str(movie_details.get('release_date', 'N/A'))
                    directors = str(movie_details.get('directors', 'N/A'))
                    actors_actresses = str(movie_details.get('actors', 'N/A'))
                    averageRating = str(movie_details['averageRating'])
                    overview = movie_details['overview']

                    st.markdown(f"""
                                <div class='head_movie'>
                                    <h2>{title}</h2>
                                    <ul style='list-style-type:none;'>
                                        <li>Runtime: <span>{runtimeMinutes} minutes</span></li>
                                        <li>Release Date: <span>{startYear}</span></li>
                                        <li>Director: <span>{directors}</span></li>
                                        <li>Actors/Actresses: <span>{actors_actresses}</span></li>
                                        <li>IMDB Rating: <span>{averageRating}</span></li>
                                    </ul>
                                    <p>
                                        <span>{overview}</span>
                                    </p>
                                </div>""",
                                unsafe_allow_html=True)



        st.markdown("""
        <h2 style='color: #FFD700; text-align: center;'>Best Matching Movies</h2>
        """, unsafe_allow_html=True)


        # Show recommendations
        cols = [None, None, None]
        for index, (title, row) in enumerate(recommendations.iterrows()):
            if index % 4 == 0:
                cols = st.columns(4) 

            with cols[index % 4]:
                with st.container(border=True):
                    if row['poster_path']== 'Unknown':
                        image_path="https://images.unsplash.com/photo-1619518594466-5bfc2dcbb83d?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    else:
                        image_path = 'https://image.tmdb.org/t/p/original/' + str(row['poster_path']) 
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)

                    runtimeMinutes = str(int(row['runtime']))
                    startYear = str(row['release_date'])
                    directors = str(row['directors'])
                    actors_actresses = str(row['actors'])
                    averageRating = str(row['averageRating'])
                    overview = row['overview']

                    st.markdown(f"""
                                <div class='blur-container'>
                                <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                                <p>
                                    <span style='color: #FFD700;'>Runtime: {runtimeMinutes}<span style='color: #FFD700;'> minutes<br>
                                    <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                    <span style='color: #FFD700;'>Director: {directors}</span><br>
                                    <!-- <span style='color: #FFD700;'>Actors: {actors_actresses}</span><br> -->
                                    <span style='color: #FFD700;'>IMDB Rating: {averageRating}</span><br>
                                </p>
                                </div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='height:5px;'>
                                </div>""",
                                unsafe_allow_html=True)
        

        if selected_movie != 'None':
            st.markdown("""
                    <h2 style='color: #FFD700; text-align: center;'>With The Same Primary Actor</h2>
                    """, unsafe_allow_html=True)
            cols = st.columns(4)
            for index, (title, row) in enumerate(actor_based_recommendations.iterrows()):
                with cols[index % 4]:
                    if row['poster_path']== 'Unknown':
                        image_path="https://images.unsplash.com/photo-1619518594466-5bfc2dcbb83d?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    else:
                        image_path = 'https://image.tmdb.org/t/p/original/' + str(row['poster_path'])
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)

                    runtimeMinutes = str(int(row['runtime']))
                    startYear = str(row['release_date'])
                    directors = str(row['directors'])
                    actors_actresses = str(row['actors'])
                    averageRating = str(row['averageRating'])
                    overview = row['overview']

                    st.markdown(f"""
                                <div class='blur-container'>
                                <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                                <p>
                                    <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                    <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                    <span style='color: #FFD700;'>Director: {directors}</span><br>
                                    <span style='color: #FFD700;'>IMDB Rating: {averageRating}</span><br>
                                </p>
                                </div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)

            st.markdown("""
                <h2 style='color: #FFD700; text-align: center;'>With The Same Director</h2>
                """, unsafe_allow_html=True)

            cols = st.columns(4)
            for index, (title, row) in enumerate(director_based_recommendations.iterrows()):
                with cols[index % 4]:
                    if row['poster_path']== 'Unknown':
                        image_path="https://images.unsplash.com/photo-1619518594466-5bfc2dcbb83d?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    else:
                        image_path = 'https://image.tmdb.org/t/p/original/' + str(row['poster_path'])               
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)

                    runtimeMinutes = str(int(row['runtime']))
                    startYear = str(row['release_date'])
                    directors = str(row['directors'])
                    actors_actresses = str(row['actors'])
                    averageRating = str(row['averageRating'])
                    overview = row['overview']

                    st.markdown(f"""
                                <div class='blur-container'>
                                <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                                <p>
                                    <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                    <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                    <span style='color: #FFD700;'>Director: {directors}</span><br>
                                    <span style='color: #FFD700;'>IMDB Rating: {averageRating}</span><br>
                                </p>
                                </div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)

            st.markdown("""
                <h2 style='color: #FFD700; text-align: center;'>Future Releases</h2>
                """, unsafe_allow_html=True)
        
    
            cols = st.columns(4)
            for index, (title, row) in enumerate(unreleased_recommendations.iterrows()):
                with cols[index % 4]:
                    if row['poster_path'] == 'Unknown':
                        image_path="https://images.unsplash.com/photo-1619518594466-5bfc2dcbb83d?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    else:
                        image_path = 'https://image.tmdb.org/t/p/original/' + str(row['poster_path'])               
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)

                    runtimeMinutes = str(int(row['runtime']))
                    startYear = str(row['release_date'])
                    directors = str(row['directors'])
                    actors_actresses = str(row['actors'])
                    averageRating = str(row['averageRating'])
                    overview = row['overview']

                    st.markdown(f"""
                                <div class='blur-container'>
                                <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                                <p>
                                    <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                    <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                    <span style='color: #FFD700;'>Director: {directors}</span><br>
                                    <span style='color: #FFD700;'>IMDB Rating: {averageRating}</span><br>
                                </p>
                                </div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)
        if selected_movie == 'None':

            st.markdown("""
                <h2 style='color: #FFD700; text-align: center;'>New Popular Releases</h2>
                """, unsafe_allow_html=True)       
            cols = st.columns(4)
            for index, (title, row) in enumerate(df_new_releases.iterrows()):
                with cols[index % 4]:
                    if row['poster_path']== 'Unknown':
                        image_path="https://images.unsplash.com/photo-1619518594466-5bfc2dcbb83d?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                    else:
                        image_path = 'https://image.tmdb.org/t/p/original/' + str(row['poster_path'])               
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)


                    runtimeMinutes = str(int(row['runtime']))
                    startYear = str(row['release_date'])
                    directors = str(row['directors'])
                    actors_actresses = str(row['actors'])
                    averageRating = str(row['averageRating'])
                    overview = row['overview']

                    st.markdown(f"""
                                <div class='blur-container'>
                                <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                                <p>
                                    <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                    <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                    <span style='color: #FFD700;'>Director: {directors}</span><br>
                                    <span style='color: #FFD700;'>IMDB Rating: {averageRating}</span><br>
                                </p>
                                </div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)



            st.markdown("""
                <h2 style='color: #FFD700; text-align: center;'>Future Releases</h2>
                """, unsafe_allow_html=True)       
            cols = st.columns(4)
            for index, (title, row) in enumerate(unreleased_recommendations.iterrows()):
                with cols[index % 4]:
                    image_path = 'https://image.tmdb.org/t/p/original/' + row['poster_path']
                    html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                    st.markdown(html_image, unsafe_allow_html=True)

                    runtimeMinutes = str(int(row['runtime']))
                    startYear = str(row['release_date'])
                    directors = str(row['directors'])
                    actors_actresses = str(row['actors'])
                    averageRating = str(row['averageRating'])
                    overview = row['overview']

                    st.markdown(f"""
                                <div class='blur-container'>
                                <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                                <p>
                                    <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                    <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                    <span style='color: #FFD700;'>Director: {directors}</span><br>
                                    <span style='color: #FFD700;'>IMDB Rating: {averageRating}</span><br>
                                </p>
                                </div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)

