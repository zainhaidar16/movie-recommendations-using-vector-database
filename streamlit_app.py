from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline


INDEX_VERSION = "2"
CHUNK_SIZE = 1000


def _load_movies() -> pd.DataFrame:
    csv_path = Path("movies.csv")
    if not csv_path.exists():
        raise FileNotFoundError("movies.csv not found. Run fetch_movies.py first.")

    df = pd.read_csv(csv_path).reset_index(drop=True)
    for column in ["title", "overview", "genre", "poster_path"]:
        df[column] = df[column].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).astype(float)
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").fillna(0).astype(int)
    return df


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_to_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "movie_id": int(row["movie_id"]),
        "title": str(row["title"]),
        "overview": str(row["overview"]),
        "genre": str(row["genre"]),
        "year": int(row["year"]),
        "rating": float(row["rating"]),
        "poster_path": str(row["poster_path"]),
    }


def _metadata_from_chroma(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_python_scalar(value) for key, value in metadata.items()}


def _genre_matches(metadata: dict[str, Any], selected_genre: str) -> bool:
    if not selected_genre or selected_genre == "All genres":
        return True
    return selected_genre.lower() in str(metadata.get("genre", "")).lower()


def _passes_filters(
    metadata: dict[str, Any],
    rating_range: tuple[float, float],
    year_range: tuple[int, int],
    selected_genre: str,
) -> bool:
    rating = float(metadata.get("rating", 0.0))
    year = int(metadata.get("year", 0))
    return (
        rating_range[0] <= rating <= rating_range[1]
        and year_range[0] <= year <= year_range[1]
        and _genre_matches(metadata, selected_genre)
    )


def _build_where_clause(rating_range: tuple[float, float], year_range: tuple[int, int]) -> dict[str, Any]:
    return {
        "$and": [
            {"rating": {"$gte": float(rating_range[0])}},
            {"rating": {"$lte": float(rating_range[1])}},
            {"year": {"$gte": int(year_range[0])}},
            {"year": {"$lte": int(year_range[1])}},
        ]
    }


def _encode_clip(model: SentenceTransformer, payload: Any) -> np.ndarray:
    embeddings = model.encode(
        [payload],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vector = np.asarray(embeddings)
    if vector.ndim == 2:
        return vector[0]
    return vector


def _chunked(sequence: list[Any], chunk_size: int) -> list[list[Any]]:
    return [sequence[index : index + chunk_size] for index in range(0, len(sequence), chunk_size)]


@st.cache_resource
def get_clip_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("clip-ViT-B-32", device=device)


@st.cache_resource
def build_indexes(csv_signature: float):
    df = _load_movies()
    clip_model = get_clip_model()
    client = chromadb.PersistentClient(path="./chroma_movies")
    try:
        text_collection = client.get_collection(name="movies_text")
    except Exception:
        text_collection = client.get_or_create_collection(
            name="movies_text",
            metadata={"hnsw:space": "cosine", "app_version": INDEX_VERSION},
        )
    else:
        if (getattr(text_collection, "metadata", None) or {}).get("app_version") != INDEX_VERSION:
            client.delete_collection(name="movies_text")
            text_collection = client.get_or_create_collection(
                name="movies_text",
                metadata={"hnsw:space": "cosine", "app_version": INDEX_VERSION},
            )

    try:
        image_collection = client.get_collection(name="movies_image")
    except Exception:
        image_collection = client.get_or_create_collection(
            name="movies_image",
            metadata={"hnsw:space": "cosine", "app_version": INDEX_VERSION},
        )
    else:
        if (getattr(image_collection, "metadata", None) or {}).get("app_version") != INDEX_VERSION:
            client.delete_collection(name="movies_image")
            image_collection = client.get_or_create_collection(
                name="movies_image",
                metadata={"hnsw:space": "cosine", "app_version": INDEX_VERSION},
            )

    progress_placeholder = None
    progress_bar = None

    if text_collection.count() == 0 or image_collection.count() == 0:
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Indexing movies...")

    if text_collection.count() == 0:
        text_payloads = [
            f"{title}. {overview}. {genre}."
            for title, overview, genre in zip(df["title"].tolist(), df["overview"].tolist(), df["genre"].tolist())
        ]
        text_embeddings = np.asarray(
            clip_model.encode(
                text_payloads,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )
        text_ids = [str(index) for index in df.index]
        text_documents = df["overview"].tolist()
        text_metadatas = [_row_to_metadata(row) for _, row in df.iterrows()]
        for chunk_index, start in enumerate(range(0, len(text_ids), CHUNK_SIZE), start=1):
            end = start + CHUNK_SIZE
            text_collection.add(
                ids=text_ids[start:end],
                embeddings=[embedding.tolist() for embedding in text_embeddings[start:end]],
                documents=text_documents[start:end],
                metadatas=text_metadatas[start:end],
            )
            if progress_bar is not None:
                progress = min(50, int((chunk_index / max(1, (len(text_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE)) * 50))
                progress_bar.progress(progress, text="Indexing movie text embeddings...")

    if image_collection.count() == 0:
        image_rows = []
        for index, row in df.iterrows():
            poster_path = Path(row["poster_path"])
            if not poster_path.exists():
                print(f"Skipping missing poster for row {index}: {poster_path}")
                continue
            try:
                with Image.open(poster_path) as image:
                    image_rows.append((index, row, image.convert("RGB")))
            except Exception as exc:
                print(f"Skipping unreadable poster for row {index}: {poster_path} ({exc})")
                continue

        if image_rows:
            image_ids = [str(index) for index, _, _ in image_rows]
            image_metadatas = [_row_to_metadata(row) for _, row, _ in image_rows]
            image_documents = [row["overview"] for _, row, _ in image_rows]
            image_inputs = [image for _, _, image in image_rows]
            image_embeddings = np.asarray(
                clip_model.encode(
                    image_inputs,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
            )
            total_image_chunks = max(1, (len(image_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE)
            for chunk_index, start in enumerate(range(0, len(image_ids), CHUNK_SIZE), start=1):
                end = start + CHUNK_SIZE
                image_collection.add(
                    ids=image_ids[start:end],
                    embeddings=[embedding.tolist() for embedding in image_embeddings[start:end]],
                    documents=image_documents[start:end],
                    metadatas=image_metadatas[start:end],
                )
                if progress_bar is not None:
                    progress = 50 + int((chunk_index / total_image_chunks) * 50)
                    progress_bar.progress(min(progress, 100), text="Indexing poster embeddings...")

    bm25 = BM25Okapi([_tokenize(text) for text in df["overview"].tolist()])
    indexed_poster_count = sum(1 for path in df["poster_path"].tolist() if Path(path).exists())
    if progress_placeholder is not None:
        progress_placeholder.empty()
    return {
        "df": df,
        "client": client,
        "text_collection": text_collection,
        "image_collection": image_collection,
        "bm25": bm25,
        "indexed_poster_count": indexed_poster_count,
        "csv_signature": csv_signature,
    }


def _query_collection(
    collection,
    query_embedding: np.ndarray,
    rating_range: tuple[float, float],
    year_range: tuple[int, int],
    selected_genre: str,
    k: int,
) -> list[tuple[int, dict[str, Any], float]]:
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=max(k * 4, 20),
        where=_build_where_clause(rating_range, year_range),
        include=["metadatas", "distances"],
    )

    output: list[tuple[int, dict[str, Any], float]] = []
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for result_id, metadata, distance in zip(ids, metadatas, distances):
        normalized_metadata = _metadata_from_chroma(metadata)
        if not _genre_matches(normalized_metadata, selected_genre):
            continue
        output.append((int(result_id), normalized_metadata, float(1.0 - distance)))
        if len(output) >= k:
            break
    return output


def vector_search(
    collection,
    query_text: str,
    model: SentenceTransformer,
    rating_range: tuple[float, float],
    year_range: tuple[int, int],
    selected_genre: str,
    k: int = 5,
) -> list[tuple[int, dict[str, Any], float]]:
    query_text = query_text.strip()
    if not query_text:
        return []
    query_embedding = _encode_clip(model, query_text)
    return _query_collection(collection, query_embedding, rating_range, year_range, selected_genre, k)


def keyword_search(
    df: pd.DataFrame,
    bm25: BM25Okapi,
    query: str,
    rating_range: tuple[float, float],
    year_range: tuple[int, int],
    selected_genre: str,
    k: int = 5,
) -> list[tuple[int, dict[str, Any], float]]:
    query = query.strip().lower()
    if not query:
        return []

    tokens = query.split()
    scores = bm25.get_scores(tokens)
    ranked_indices = np.argsort(scores)[::-1]
    results: list[tuple[int, dict[str, Any], float]] = []

    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        row = df.iloc[int(index)]
        metadata = _row_to_metadata(row)
        if not _passes_filters(metadata, rating_range, year_range, selected_genre):
            continue
        results.append((int(index), metadata, score))
        if len(results) >= k:
            break
    return results


def hybrid_search(
    df: pd.DataFrame,
    bm25: BM25Okapi,
    collection,
    query_text: str,
    model: SentenceTransformer,
    rating_range: tuple[float, float],
    year_range: tuple[int, int],
    selected_genre: str,
    k: int = 5,
) -> list[tuple[int, dict[str, Any], float]]:
    vector_hits = vector_search(
        collection,
        query_text,
        model,
        rating_range,
        year_range,
        selected_genre,
        k=20,
    )
    keyword_hits = keyword_search(
        df,
        bm25,
        query_text,
        rating_range,
        year_range,
        selected_genre,
        k=20,
    )

    fused_scores: dict[int, float] = {}
    metadata_by_id: dict[int, dict[str, Any]] = {}

    for rank, (row_id, metadata, _) in enumerate(vector_hits, start=1):
        fused_scores[row_id] = fused_scores.get(row_id, 0.0) + 1.0 / (60 + rank)
        metadata_by_id.setdefault(row_id, metadata)

    for rank, (row_id, metadata, _) in enumerate(keyword_hits, start=1):
        fused_scores[row_id] = fused_scores.get(row_id, 0.0) + 1.0 / (60 + rank)
        metadata_by_id.setdefault(row_id, metadata)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(row_id, metadata_by_id[row_id], score) for row_id, score in ranked[:k]]


def image_search(
    collection,
    pil_image: Image.Image,
    model: SentenceTransformer,
    rating_range: tuple[float, float],
    year_range: tuple[int, int],
    selected_genre: str,
    k: int = 5,
) -> list[tuple[int, dict[str, Any], float]]:
    if collection.count() == 0:
        return []
    query_embedding = _encode_clip(model, pil_image.convert("RGB"))
    return _query_collection(collection, query_embedding, rating_range, year_range, selected_genre, k)


@st.cache_resource
def get_rag_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base", device=-1)


def _render_movie_result(row_id: int, metadata: dict[str, Any], score: float):
    with st.expander(f"{metadata['title']} ({metadata['year']})"):
        poster_path = Path(metadata["poster_path"])
        if poster_path.exists():
            st.image(str(poster_path), width=220)
        st.write(f"**Title:** {metadata['title']}")
        st.write(f"**Year:** {metadata['year']}")
        st.write(f"**Genre:** {metadata['genre']}")
        st.write(f"**Rating:** {metadata['rating']}")
        st.write(f"**Score:** {score:.4f}")
        st.write(metadata["overview"])
        st.caption(f"Row ID: {row_id}")


def _get_year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    years = df["year"].tolist()
    return int(min(years)), int(max(years))


def _get_rating_bounds(df: pd.DataFrame) -> tuple[float, float]:
    ratings = df["rating"].tolist()
    return float(min(ratings)), float(max(ratings))


def main():
    st.set_page_config(page_title="Movie Semantic Search", layout="wide")
    st.title("Movie Semantic Search")

    try:
        csv_signature = Path("movies.csv").stat().st_mtime if Path("movies.csv").exists() else 0.0
        state = build_indexes(csv_signature)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df = state["df"]
    bm25 = state["bm25"]
    text_collection = state["text_collection"]
    image_collection = state["image_collection"]
    clip_model = get_clip_model()
    rating_bounds = _get_rating_bounds(df)
    year_bounds = _get_year_bounds(df)
    genre_options = ["All genres"] + sorted({genre for genre in df["genre"].tolist() if genre})

    text_tab, image_tab, details_tab, rag_tab = st.tabs(["Text Search", "Image Search", "Movie Details", "Recommend (RAG)"])

    with text_tab:
        st.subheader("Text Search")
        query_text = st.text_input("Search query", placeholder="mind-bending sci-fi about dreams")
        search_method = st.radio("Method", ["Vector", "Keyword", "Hybrid"], horizontal=True)
        vector_target = st.radio("Vector matches against", ["text", "image"], horizontal=True)
        rating_range = st.slider("Rating range", min_value=rating_bounds[0], max_value=rating_bounds[1], value=rating_bounds, step=0.1)
        year_range = st.slider("Year range", min_value=year_bounds[0], max_value=year_bounds[1], value=year_bounds, step=1)
        selected_genre = st.selectbox("Genre", genre_options)

        if query_text.strip():
            target_collection = text_collection if vector_target == "text" else image_collection
            if search_method == "Vector":
                results = vector_search(target_collection, query_text, clip_model, rating_range, year_range, selected_genre)
            elif search_method == "Keyword":
                results = keyword_search(df, bm25, query_text, rating_range, year_range, selected_genre)
            else:
                results = hybrid_search(df, bm25, target_collection, query_text, clip_model, rating_range, year_range, selected_genre)

            if results:
                for row_id, metadata, score in results:
                    _render_movie_result(row_id, metadata, score)
            else:
                st.info("No results matched the current filters.")

    with image_tab:
        st.subheader("Image Search")
        uploaded = st.file_uploader("Upload a poster image", type=["png", "jpg", "jpeg", "webp"])
        rating_range = st.slider(
            "Rating range",
            min_value=rating_bounds[0],
            max_value=rating_bounds[1],
            value=rating_bounds,
            step=0.1,
            key="image_rating_range",
        )
        year_range = st.slider(
            "Year range",
            min_value=year_bounds[0],
            max_value=year_bounds[1],
            value=year_bounds,
            step=1,
            key="image_year_range",
        )
        selected_genre = st.selectbox("Genre", genre_options, key="image_genre")

        if image_collection.count() == 0 or state["indexed_poster_count"] == 0:
            st.warning("No posters are indexed yet, so image search cannot return results.")
        elif uploaded is not None:
            query_image = Image.open(uploaded).convert("RGB")
            st.image(query_image, caption="Query image", width=260)
            results = image_search(image_collection, query_image, clip_model, rating_range, year_range, selected_genre)
            if results:
                for row_id, metadata, score in results:
                    _render_movie_result(row_id, metadata, score)
            else:
                st.info("No visually similar movies matched the current filters.")

    with details_tab:
        st.subheader("Movie Details")
        row_id = st.number_input("Enter a row id", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
        if len(df) > 0:
            selected_row = df.iloc[int(row_id)]
            metadata = _row_to_metadata(selected_row)
            poster_path = Path(metadata["poster_path"])
            cols = st.columns([1, 1.5])
            with cols[0]:
                if poster_path.exists():
                    st.image(str(poster_path), width="stretch")
                else:
                    st.warning("Poster file is missing locally.")
            with cols[1]:
                st.write(f"**Title:** {metadata['title']}")
                st.write(f"**Movie ID:** {metadata['movie_id']}")
                st.write(f"**Genre:** {metadata['genre']}")
                st.write(f"**Year:** {metadata['year']}")
                st.write(f"**Rating:** {metadata['rating']}")
                st.write("**Overview:**")
                st.write(metadata["overview"])

    with rag_tab:
        st.subheader("Recommend (RAG)")
        request_text = st.text_input("What kind of movie do you want?", placeholder="thought-provoking sci-fi thriller")
        occasion = st.text_input("Occasion", value="any occasion")

        if request_text.strip() and occasion.strip():
            rag_sources = vector_search(
                text_collection,
                request_text,
                clip_model,
                rating_bounds,
                year_bounds,
                "All genres",
                k=3,
            )
            if rag_sources:
                source_lines = []
                for index, (_, metadata, _) in enumerate(rag_sources, start=1):
                    source_lines.append(f"{index}. Title: {metadata['title']}\nOverview: {metadata['overview']}")

                prompt = (
                    f"The user wants {request_text} for {occasion}.\n\n"
                    "Use these candidate movies as context and recommend one best match.\n\n"
                    + "\n\n".join(source_lines)
                    + "\n\nAnswer with a short recommendation and why it fits."
                )

                with st.spinner("Generating recommendation..."):
                    generator = get_rag_pipeline()
                    generated = generator(
                        prompt,
                        max_new_tokens=128,
                        do_sample=False,
                        truncation=True,
                    )[0]["generated_text"]

                st.write(generated)
                st.markdown("**Sources used**")
                for row_id, metadata, score in rag_sources:
                    _render_movie_result(row_id, metadata, score)
            else:
                st.info("No source movies were found for the recommendation prompt.")


if __name__ == "__main__":
    main()
