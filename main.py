import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,QuantileTransformer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Load data
movies_path = r"C:\Users\Admin\Desktop\Movie\data\df_cinema2.csv"
df_movies = pd.read_csv(movies_path, low_memory=False)

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
    genre_options = np.sort(df_movies['tmdb_genres'].str.split(',', expand=True).stack().unique(), 0)
    selected_genres = st.multiselect("", options=genre_options)

col5, col6 = st.columns([1, 3])
with col5:
    st.markdown("<br><span class='label'> Select actor(s) (optional):</span>", unsafe_allow_html=True)
with col6:
    actor_options = np.sort(np.unique(df_movies['actors'].str.split(',', expand=True).stack()))
    selected_actors = st.multiselect("", options=actor_options)

col7, col8 = st.columns([1, 3])
with col7:
    st.markdown("<br><span class='label'> Select the number of recommendations:</span>", unsafe_allow_html=True)
with col8:
    num_recommendations = st.slider("", 1, 20, 20)


# Function to put weight on genre or actor features if the user selects them
def create_dummy_columns(df, names_list, column_name, df_column):
    """Create dummy columns for categorical variables."""
    for name in names_list:
        if name:
            name = name.strip().lower()
            df[column_name + name] = df[df_column].apply(lambda x: 1 if isinstance(x, str) and name in x.lower() else 0)




# The main function to make recommendations
def get_recommendations(movie_title, selected_genres, selected_actors, num_recs):
    df = df_movies.copy()  # Work on a copy to avoid changing the original dataframe

    # Re-include the selected movie if it's been filtered out
    if movie_title != 'None':
        selected_movie_df = df_movies[df_movies['title'] == movie_title]
        df = pd.concat([df, selected_movie_df], ignore_index=True)

    if movie_title != 'None':
        selected_movie_df = df[df['title'] == movie_title].iloc[0]
        index_movie = df[df['title'] == movie_title].index[0]
        # Ensure the split operation is safe
        actors_in_chosen_film = set(selected_movie_df['actors'].split(", ")[:6])
        directors_in_chosen_film = selected_movie_df['directors'].split(", ")
        writers_in_chosen_film = selected_movie_df['writers'].split(", ")[:2]
        genres = selected_movie_df['tmdb_genres'].split(",")
        prod_companies = selected_movie_df['production_companies'].split(", ")[:1]
        title_words = selected_movie_df['title'].split(" ")
        important_actors_in_chosen_film = set(selected_movie_df['actors'].split(", ")[:4])
        lead_actors_in_chosen_film = set(selected_movie_df['actors'].split(", ")[:2])
        keywords_in_chosen_movie=selected_movie_df['keywords'].split(", ")

        # Apply weights to dummy columns
        create_dummy_columns(df, actors_in_chosen_film, 'actor_', 'actors')
        create_dummy_columns(df, directors_in_chosen_film, 'director_', 'directors')
        create_dummy_columns(df, writers_in_chosen_film, 'writer_', 'writers')
        create_dummy_columns(df, genres, 'genre_', 'tmdb_genres')
        create_dummy_columns(df, prod_companies, 'production_company_', 'production_companies')
        create_dummy_columns(df, title_words, 'tile_word_', 'cleaned_title')
        #create_dummy_columns(df, important_actors_in_chosen_film, 'lead_actor_', 'actors')
        create_dummy_columns(df, keywords_in_chosen_movie, 'keyword_', 'keywords')

        
        
        # Making knn df with only numerical cols
        X = df.drop(columns=['year','numVotes','vote_count','budget','revenue','imdb_rating','runtime','keywords','imdb_rating','tmdb_genres', 'overview', 'poster_path', 'title', 'actors', 'directors', 'writers',  "production_companies"])
        X = X.select_dtypes(include=np.number)  # Ensure only numeric columns are selected
        # Scaling
        #scaler= RobustScaler()
        #scaler=StandardScaler()
        #scaler = MinMaxScaler()
        scaler= QuantileTransformer(output_distribution='normal')
        X_scaled = scaler.fit_transform(X)
        # Fit model with user specified number of neighbors
        model = NearestNeighbors(n_neighbors=len(df), algorithm='auto', metric='cosine')
        model.fit(X)

        distances, recommendation_indices = model.kneighbors(X=[X_scaled[index_movie]], return_distance=True)

        # Retrieve the recommended movies based on the indices
        recommended_movies = df.iloc[recommendation_indices[0]]

        # Add the distances as a new column in the recommended_movies DataFrame
        recommended_movies['distance'] = distances[0]

        if selected_genres:
            recommended_movies['genre_matches'] = recommended_movies['tmdb_genres'].apply(lambda x: sum(genre.strip() in [g.strip() for g in x.split(",")] for genre in selected_genres))
            recommended_movies['distance'] = recommended_movies.apply(lambda row: row['distance'] * (0.7 ** row['genre_matches']), axis=1)
        
        # Sort the recommended movies by the adjusted distance
            recommended_movies = recommended_movies.sort_values(by='distance')
        
        if selected_actors:
            recommended_movies['actor_matches'] = recommended_movies['actors'].apply(lambda x: sum(genre.strip() in [g.strip() for g in x.split(",")] for genre in selected_actors))
            recommended_movies['distance'] = recommended_movies.apply(lambda row: row['distance'] * (0.6 ** row['actor_matches']), axis=1)
            create_dummy_columns(df, lead_actors_in_chosen_film, 'lead_actor_', 'actors')
            create_dummy_columns(df, lead_actors_in_chosen_film, 'lead_actor_', 'actors')
            recommended_movies = recommended_movies.sort_values(by='distance')

            

        recommended_movies['unadjusted_distance'] = distances[0]
        return recommended_movies.set_index('title')[['runtime', 'tmdb_genres', 'year', 'imdb_rating','actors', 'directors', 'overview', 'poster_path']]
        

    else:
        recommended_movies = df.copy()
        recommended_movies['genre_matches'] = 0
        recommended_movies['actor_matches'] = 0

        if selected_genres:
            selected_genres_lower = [genre.strip().lower() for genre in selected_genres]
            recommended_movies['genre_matches'] = recommended_movies['tmdb_genres'].apply(
                lambda x: sum(genre in [g.strip().lower() for g in x.split(",")] for genre in selected_genres_lower)
            )

        if selected_genres:
            selected_genres_lower = [genre.strip().lower() for genre in selected_genres]
            recommended_movies['genre_matches'] = recommended_movies['tmdb_genres'].apply(
            lambda x: sum(genre in [g.strip().lower() for g in x.split(",")] for genre in selected_genres_lower)
            )



        recommended_movies['genre_actor_match_count'] = recommended_movies['actor_matches'] + recommended_movies['genre_matches']
        recommended_movies = recommended_movies.sort_values(by=["genre_actor_match_count", "imdb_rating"], ascending=[False, False])
        recommended_movies = recommended_movies.head(num_recommendations)


         
    return recommended_movies.set_index('title')[['runtime', 'tmdb_genres', 'year', 'imdb_rating', 'actors', 'directors', 'overview', 'poster_path']]


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
    # Fetch recommendations, potentially with one extra to exclude the selected movie later
    adjusted_num_recs = num_recommendations
    recommendations = get_recommendations(selected_movie, selected_genres, selected_actors, adjusted_num_recs)
    if selected_movie != 'None':
        movie_details = df_movies[df_movies['title'] == selected_movie].iloc[0]
        first_actor = movie_details['actors'].split(',')[:2] if movie_details['actors'] else None
        first_director = movie_details['directors'].split(',')[0] if movie_details['directors'] else None 
        main_recommendations = get_recommendations(selected_movie, selected_genres, selected_actors, adjusted_num_recs)
        selected_movie_row = df_movies[df_movies['title'] == selected_movie]
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
                image_path = 'https://image.tmdb.org/t/p/original/' + movie_details['poster_path']
                html_image = f"<img src='{image_path}' alt='Image' style='width: 75%; border-radius:14px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                st.markdown(html_image, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: px;'></div>", unsafe_allow_html=True)

            with col2:
                title = str(movie_details['title'])
                runtimeMinutes = str(int(movie_details['runtime']))
                startYear = str(movie_details.get('year', 'N/A'))
                directors = str(movie_details.get('directors', 'N/A'))
                actors_actresses = str(movie_details.get('actors', 'N/A'))
                averageRating = str(round(movie_details['imdb_rating'], 2))
                overview = movie_details['overview']

                st.markdown(f"""
                            <div class='head_movie'>
                                <h2>{title}</h2>
                                <ul style='list-style-type:none;'>
                                    <li>Runtime: <span>{runtimeMinutes} minutes</span></li>
                                    <li>Release Year: <span>{startYear}</span></li>
                                    <li>Director: <span>{directors}</span></li>
                                    <li>Actors/Actresses: <span>{actors_actresses}</span></li>
                                    <li>Average Rating: <span>{averageRating}</span></li>
                                </ul>
                                <p>
                                    <span>{overview}</span>
                                </p>
                            </div>""",
                            unsafe_allow_html=True)
            # Exclude the selected movie from recommendations if it's in the list
            recommendations = recommendations[recommendations.index != selected_movie]


    # Only the requested number of recommendations
    recommendations = recommendations.head(adjusted_num_recs)

    st.markdown("""
    <h2 style='color: #FFD700; text-align: center;'>Recommendations</h2>
    """, unsafe_allow_html=True)
    # Show recommendations
    cols = [None, None, None]
    for index, (title, row) in enumerate(recommendations.iterrows()):
        if index % 4 == 0:
            cols = st.columns(4) 

        with cols[index % 4]:
            with st.container(border=True):
                image_path = 'https://image.tmdb.org/t/p/original/'+row['poster_path']
                html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                st.markdown(html_image, unsafe_allow_html=True)

                runtimeMinutes = str(int(row['runtime']))
                startYear = str(row['year'])
                directors = str(row['directors'])
                actors_actresses = str(row['actors'])
                averageRating = str(round(row['imdb_rating'],2))
                overview = row['overview']

                st.markdown(f"""
                            <div class='blur-container'>
                            <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                            <p>
                                <span style='color: #FFD700;'>Runtime: {runtimeMinutes}<span style='color: #FFD700;'> minutes<br>
                                <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                <span style='color: #FFD700;'>Director: {directors}</span><br>
                                <!-- <span style='color: #FFD700;'>Actors: {actors_actresses}</span><br> -->
                                <span style='color: #FFD700;'>Rating: {averageRating}</span><br>
                            </p>
                            </div>""",
                            unsafe_allow_html=True)

                st.markdown(f"""<div style='height:5px;'>
                            </div>""",
                            unsafe_allow_html=True)
    

    if selected_movie != 'None':
        st.markdown("""
                <h2 style='color: #FFD700; text-align: center;'>Recommendations With Primary Actors</h2>
                """, unsafe_allow_html=True)
        cols = st.columns(4)
        for index, (title, row) in enumerate(actor_based_recommendations.iterrows()):
            with cols[index % 4]:
                image_path = 'https://image.tmdb.org/t/p/original/' + row['poster_path']
                html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                st.markdown(html_image, unsafe_allow_html=True)

                runtimeMinutes = str(int(row['runtime']))
                startYear = str(row['year'])
                directors = str(row['directors'])
                actors_actresses = str(row['actors'])
                averageRating = str(round(row['imdb_rating'], 2))
                overview = row['overview']

                st.markdown(f"""
                            <div class='blur-container'>
                            <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                            <p>
                                <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                <span style='color: #FFD700;'>Director: {directors}</span><br>
                                <span style='color: #FFD700;'>Rating: {averageRating}</span><br>
                            </p>
                            </div>""",
                            unsafe_allow_html=True)

                st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)

        st.markdown("""
            <h2 style='color: #FFD700; text-align: center;'>Recommendations With Director</h2>
            """, unsafe_allow_html=True)

        cols = st.columns(4)
        for index, (title, row) in enumerate(director_based_recommendations.iterrows()):
            with cols[index % 4]:
                image_path = 'https://image.tmdb.org/t/p/original/' + row['poster_path']
                html_image = f"<img src='{image_path}' alt='Image' style='width: 100%; border-radius:8px;box-shadow: 1px 1px 15px 1px #c4c4c4;'>"
                st.markdown(html_image, unsafe_allow_html=True)

                runtimeMinutes = str(int(row['runtime']))
                startYear = str(row['year'])
                directors = str(row['directors'])
                actors_actresses = str(row['actors'])
                averageRating = str(round(row['imdb_rating'], 2))
                overview = row['overview']

                st.markdown(f"""
                            <div class='blur-container'>
                            <h2 style='color: #FFD700;'>{title}</h2> <!-- Title color -->
                            <p>
                                <span style='color: #FFD700;'>Runtime: {runtimeMinutes} minutes</span><br>
                                <span style='color: #FFD700;'>Year: {startYear}</span><br>
                                <span style='color: #FFD700;'>Director: {directors}</span><br>
                                <span style='color: #FFD700;'>Rating: {averageRating}</span><br>
                            </p>
                            </div>""",
                            unsafe_allow_html=True)

                st.markdown(f"""<div style='height:5px;'></div>""", unsafe_allow_html=True)   
