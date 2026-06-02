import csv
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv


BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
PAGES = int(os.getenv("PAGES", "15"))
OUTPUT_COLUMNS = [
    "movie_id",
    "title",
    "overview",
    "genre",
    "year",
    "rating",
    "poster_path",
]


def _get_json(session: requests.Session, endpoint: str, api_key: str, **params):
    response = session.get(
        f"{BASE_URL}{endpoint}",
        params={"api_key": api_key, **params},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _download_poster(session: requests.Session, poster_path: str, movie_id: int) -> Optional[str]:
    posters_dir = Path("posters")
    posters_dir.mkdir(parents=True, exist_ok=True)
    local_path = posters_dir / f"{movie_id}.jpg"
    if local_path.exists():
        return f"posters/{movie_id}.jpg"

    poster_url = f"{IMAGE_BASE_URL}{poster_path}"
    response = session.get(poster_url, timeout=30)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    return f"posters/{movie_id}.jpg"


def main():
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is missing from .env")

    session = requests.Session()
    genre_payload = _get_json(session, "/genre/movie/list", api_key)
    genre_map = {item["id"]: item["name"] for item in genre_payload.get("genres", [])}

    rows = []
    for page in range(1, PAGES + 1):
        payload = _get_json(session, "/movie/top_rated", api_key, page=page)
        movies = payload.get("results", [])
        kept_this_page = 0

        for movie in movies:
            overview = (movie.get("overview") or "").strip()
            poster_path = movie.get("poster_path") or ""
            release_date = (movie.get("release_date") or "").strip()
            if not overview or not poster_path or len(release_date) < 4:
                continue

            try:
                local_poster_path = _download_poster(session, poster_path, int(movie["id"]))
            except requests.RequestException as exc:
                print(f"Skipping movie {movie.get('id')} because poster download failed: {exc}")
                continue

            genre_names = [genre_map.get(genre_id, "") for genre_id in movie.get("genre_ids", [])]
            genre = "/".join([genre for genre in genre_names if genre])

            rows.append(
                {
                    "movie_id": int(movie["id"]),
                    "title": (movie.get("title") or movie.get("original_title") or "").strip(),
                    "overview": overview,
                    "genre": genre,
                    "year": int(release_date[:4]),
                    "rating": float(movie.get("vote_average") or 0.0),
                    "poster_path": local_poster_path,
                }
            )
            kept_this_page += 1

        print(f"Page {page}/{PAGES}: kept {kept_this_page} movies")
        if page < PAGES:
            time.sleep(0.3)

    with Path("movies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Finished. Wrote {len(rows)} movies to movies.csv")


if __name__ == "__main__":
    main()
