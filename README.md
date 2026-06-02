# Movie Semantic Search

## Run

1. Put your TMDB v3 API key in `.env` as `TMDB_API_KEY`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Build the dataset with `python fetch_movies.py`.
4. Launch the app with `streamlit run streamlit_app.py`.

The app uses local ChromaDB persistence in `./chroma_movies` and local poster files in `./posters`.
