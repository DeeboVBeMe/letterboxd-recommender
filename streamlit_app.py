#!/usr/bin/env python3
"""
Letterboxd Movie Recommender — Streamlit Web Version
With richer recommendation explanations.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RatedFilm:
    title: str
    year: Optional[int]
    user_rating: float
    letterboxd_avg: Optional[float] = None
    tmdb_id: Optional[int] = None
    genres: list[str] = field(default_factory=list)
    directors: list[str] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)

    @property
    def rating_diff(self) -> Optional[float]:
        if self.letterboxd_avg is None:
            return None
        return round(self.user_rating - self.letterboxd_avg, 2)


@dataclass
class Recommendation:
    title: str
    year: Optional[int]
    tmdb_id: int
    score: float
    reasons: list[str]
    explanation: str = ""
    tmdb_vote: Optional[float] = None
    overview: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(obj, key, default=None):
    try:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def to_list(obj):
    if obj is None:
        return []
    try:
        return list(obj)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_from_csv(uploaded_file) -> list[RatedFilm]:
    df = pd.read_csv(uploaded_file)
    colmap = {c.lower().strip(): c for c in df.columns}

    title_col = next((colmap[k] for k in ("title", "name", "film") if k in colmap), None)
    year_col = next((colmap[k] for k in ("year", "release_year") if k in colmap), None)
    rating_col = next((colmap[k] for k in ("rating", "user_rating", "stars") if k in colmap), None)

    if not title_col or not rating_col:
        raise ValueError(
            "CSV must have at least 'Title' and 'Rating' columns.\n"
            "Example:\nTitle,Year,Rating\nParasite,2019,5\nHeat,1995,4.5"
        )

    films: list[RatedFilm] = []
    for _, row in df.iterrows():
        try:
            rating = float(row[rating_col])
            if rating <= 0:
                continue
            if rating > 5:
                rating = rating / 2.0
            year = None
            if year_col and pd.notna(row.get(year_col)):
                year = int(row[year_col])
            films.append(
                RatedFilm(
                    title=str(row[title_col]).strip(),
                    year=year,
                    user_rating=rating,
                )
            )
        except (ValueError, TypeError):
            continue
    return films


# ---------------------------------------------------------------------------
# TMDB client
# ---------------------------------------------------------------------------

class TMDBClient:
    def __init__(self, api_key: str):
        from tmdbv3api import TMDb, Movie, Search, Person, Discover

        self.tmdb = TMDb()
        self.tmdb.api_key = api_key
        self.tmdb.language = "en"
        self.movie = Movie()
        self.search = Search()
        self.person_api = Person()
        self.discover = Discover()
        self._cache: dict[str, Any] = {}
        self.last_error = None

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[dict]:
        key = f"search:{title}:{year}"
        if key in self._cache:
            return self._cache[key]
        try:
            results = self.search.movies(title, year=year) if year else self.search.movies(title)
            results = to_list(results)
            if results:
                best = results[0]
                release = safe_get(best, "release_date")
                data = {
                    "id": safe_get(best, "id"),
                    "title": safe_get(best, "title"),
                    "year": int(release[:4]) if release else None,
                    "vote_average": safe_get(best, "vote_average"),
                    "overview": safe_get(best, "overview"),
                }
                self._cache[key] = data
                return data
            self.last_error = f"No results for '{title}'"
            self._cache[key] = None
            return None
        except Exception as e:
            self.last_error = str(e)
            self._cache[key] = None
            return None

    def get_details(self, tmdb_id: int) -> Optional[dict]:
        key = f"details:{tmdb_id}"
        if key in self._cache:
            return self._cache[key]
        try:
            m = self.movie.details(tmdb_id)
            credits = self.movie.credits(tmdb_id)

            genres = []
            for g in to_list(safe_get(m, "genres")):
                name = safe_get(g, "name")
                if name:
                    genres.append(name)

            directors = []
            for c in to_list(safe_get(credits, "crew")):
                if safe_get(c, "job") == "Director":
                    name = safe_get(c, "name")
                    if name:
                        directors.append(name)

            actors = []
            for c in to_list(safe_get(credits, "cast"))[:8]:
                name = safe_get(c, "name")
                if name:
                    actors.append(name)

            release = safe_get(m, "release_date")
            data = {
                "id": tmdb_id,
                "title": safe_get(m, "title"),
                "year": int(release[:4]) if release else None,
                "genres": genres,
                "directors": directors,
                "actors": actors,
                "vote_average": safe_get(m, "vote_average"),
                "overview": safe_get(m, "overview"),
            }
            self._cache[key] = data
            return data
        except Exception as e:
            self.last_error = str(e)
            self._cache[key] = None
            return None

    def get_similar(self, tmdb_id: int, limit: int = 7) -> list[dict]:
        try:
            results = to_list(self.movie.similar(tmdb_id))
            out = []
            for r in results[:limit]:
                release = safe_get(r, "release_date")
                out.append({
                    "id": safe_get(r, "id"),
                    "title": safe_get(r, "title"),
                    "year": int(release[:4]) if release else None,
                    "vote_average": safe_get(r, "vote_average"),
                    "overview": safe_get(r, "overview"),
                })
            return out
        except Exception:
            return []

    def get_recommendations(self, tmdb_id: int, limit: int = 5) -> list[dict]:
        try:
            results = to_list(self.movie.recommendations(tmdb_id))
            out = []
            for r in results[:limit]:
                release = safe_get(r, "release_date")
                out.append({
                    "id": safe_get(r, "id"),
                    "title": safe_get(r, "title"),
                    "year": int(release[:4]) if release else None,
                    "vote_average": safe_get(r, "vote_average"),
                    "overview": safe_get(r, "overview"),
                })
            return out
        except Exception:
            return []

    def discover_by(self, genre_ids: list[int], limit: int = 12) -> list[dict]:
        try:
            params = {
                "with_genres": ",".join(map(str, genre_ids)),
                "sort_by": "vote_average.desc",
                "vote_count.gte": 100,
                "page": 1,
            }
            results = to_list(self.discover.discover_movies(params))
            out = []
            for r in results[:limit]:
                release = safe_get(r, "release_date")
                out.append({
                    "id": safe_get(r, "id"),
                    "title": safe_get(r, "title"),
                    "year": int(release[:4]) if release else None,
                    "vote_average": safe_get(r, "vote_average"),
                    "overview": safe_get(r, "overview"),
                })
            return out
        except Exception:
            return []

    def genre_name_to_id(self) -> dict[str, int]:
        return {
            "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
            "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
            "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
            "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
            "Thriller": 53, "War": 10752, "Western": 37,
        }


# ---------------------------------------------------------------------------
# Analysis + richer explanations
# ---------------------------------------------------------------------------

def enrich_films(films: list[RatedFilm], tmdb: TMDBClient, max_enrich: int = 40) -> list[RatedFilm]:
    ranked = sorted(films, key=lambda f: f.user_rating, reverse=True)[:max_enrich]
    progress = st.progress(0)
    status_text = st.empty()
    success_count = 0

    for i, film in enumerate(ranked):
        status_text.text(f"Looking up: {film.title}")
        result = tmdb.search_movie(film.title, film.year)
        if result and result.get("id"):
            film.tmdb_id = result["id"]
            details = tmdb.get_details(result["id"])
            if details:
                film.genres = details.get("genres", [])
                film.directors = details.get("directors", [])
                film.actors = details.get("actors", [])
                if film.letterboxd_avg is None and details.get("vote_average"):
                    film.letterboxd_avg = round(details["vote_average"] / 2, 2)
                success_count += 1
        progress.progress((i + 1) / len(ranked))
        time.sleep(0.03)

    progress.empty()
    status_text.empty()
    st.info(f"Matched **{success_count}** films to TMDB")
    return films


def analyze_preferences(films: list[RatedFilm], min_rating: float = 4.0) -> dict[str, Any]:
    high = [f for f in films if f.user_rating >= min_rating]
    if not high:
        high = sorted(films, key=lambda f: f.user_rating, reverse=True)[:20]

    genre_scores: Counter = Counter()
    director_scores: Counter = Counter()
    actor_scores: Counter = Counter()

    for f in high:
        weight = f.user_rating
        for g in f.genres:
            genre_scores[g] += weight
        for d in f.directors:
            director_scores[d] += weight
        for a in f.actors[:5]:
            actor_scores[a] += weight

    overrated = sorted(
        [f for f in films if f.rating_diff is not None and f.rating_diff >= 0.7],
        key=lambda f: f.rating_diff or 0,
        reverse=True,
    )[:8]

    return {
        "high_rated": high,
        "top_genres": genre_scores.most_common(8),
        "top_directors": director_scores.most_common(8),
        "top_actors": actor_scores.most_common(10),
        "overrated_by_user": overrated,
        "avg_user_rating": sum(f.user_rating for f in films) / len(films) if films else 0,
        "num_rated": len(films),
        "num_high": len(high),
    }


def build_explanation(
    title: str,
    reasons: list[str],
    seed_titles: list[str],
    top_genres: list[str],
    top_directors: list[str],
    tmdb_vote: Optional[float],
) -> str:
    """Turn raw signals into a short 2–3 sentence explanation."""
    parts = []

    # Similarity / seed films
    similar_seeds = [r for r in reasons if r.startswith("Similar to your")]
    if similar_seeds:
        # Extract the movie names from the reason strings
        named = []
        for r in similar_seeds[:2]:
            # format: "Similar to your 5.0★ Movie Title"
            if "★ " in r:
                named.append(r.split("★ ", 1)[1])
        if named:
            if len(named) == 1:
                parts.append(f"It shares a strong stylistic or thematic connection with *{named[0]}*, which you rated very highly.")
            else:
                parts.append(f"It sits close to films you loved such as *{named[0]}* and *{named[1]}*.")

    # Director
    dir_reasons = [r for r in reasons if r.startswith("Directed by")]
    if dir_reasons:
        dname = dir_reasons[0].replace("Directed by ", "")
        parts.append(f"It is directed by **{dname}**, one of the filmmakers who appears most often among your highest-rated movies.")

    # Genre
    genre_reasons = [r for r in reasons if "preferred genres" in r.lower() or "Matches preferred" in r]
    if genre_reasons and top_genres:
        gshow = ", ".join(top_genres[:2])
        parts.append(f"It falls squarely in your favorite territory ({gshow}).")

    # Community quality
    if tmdb_vote and tmdb_vote >= 7.8:
        parts.append(f"It also carries a strong critical reputation (TMDB {tmdb_vote:.1f}/10).")
    elif tmdb_vote and tmdb_vote >= 7.2:
        parts.append(f"Critics and audiences rate it solidly (TMDB {tmdb_vote:.1f}/10).")

    # Fallback if we somehow have almost nothing
    if not parts:
        if seed_titles:
            parts.append(f"It was recommended because of its closeness to films you already rate highly, such as *{seed_titles[0]}*.")
        else:
            parts.append("It matches several patterns in your highest-rated films.")

    # Keep it to roughly 2–3 sentences
    text = " ".join(parts[:3])
    return text


def generate_recommendations(
    films: list[RatedFilm],
    prefs: dict[str, Any],
    tmdb: TMDBClient,
    top_n: int = 15,
) -> list[Recommendation]:
    already_seen = {f.title.lower() for f in films}
    candidates: dict[int, Recommendation] = {}

    # Keep track of which of the user's films triggered each candidate
    seed_map: dict[int, list[str]] = {}

    def add_candidate(item: dict, base_score: float, reason: str, seed_title: str = ""):
        tid = item.get("id")
        title = item.get("title")
        if not tid or not title or title.lower() in already_seen:
            return
        if tid in candidates:
            candidates[tid].score += base_score
            if reason not in candidates[tid].reasons:
                candidates[tid].reasons.append(reason)
            if seed_title and seed_title not in seed_map.get(tid, []):
                seed_map.setdefault(tid, []).append(seed_title)
        else:
            candidates[tid] = Recommendation(
                title=title,
                year=item.get("year"),
                tmdb_id=tid,
                score=base_score,
                reasons=[reason],
                tmdb_vote=item.get("vote_average"),
                overview=item.get("overview"),
            )
            if seed_title:
                seed_map[tid] = [seed_title]

    high = prefs["high_rated"][:12]
    top_genre_names = [g for g, _ in prefs["top_genres"][:3]]
    top_director_names = [d for d, _ in prefs["top_directors"][:4]]

    for f in high:
        if not f.tmdb_id:
            continue
        for sim in tmdb.get_similar(f.tmdb_id, limit=6):
            add_candidate(
                sim,
                3.0 * (f.user_rating / 5),
                f"Similar to your {f.user_rating}★ {f.title}",
                seed_title=f.title,
            )
        for rec in tmdb.get_recommendations(f.tmdb_id, limit=4):
            add_candidate(
                rec,
                2.5 * (f.user_rating / 5),
                f"Recommended alongside {f.title}",
                seed_title=f.title,
            )

    genre_map = tmdb.genre_name_to_id()
    top_genre_ids = [genre_map[g] for g in top_genre_names if g in genre_map]
    if top_genre_ids:
        for disco in tmdb.discover_by(top_genre_ids, limit=12):
            add_candidate(
                disco,
                2.0,
                f"Matches preferred genres ({', '.join(top_genre_names)})",
            )

    for dname in top_director_names:
        try:
            person_search = to_list(tmdb.search.people(dname))
            if person_search:
                pid = safe_get(person_search[0], "id")
                credits = tmdb.person_api.movie_credits(pid)
                crew = to_list(safe_get(credits, "crew"))
                directed = [c for c in crew if safe_get(c, "job") == "Director"][:5]
                for c in directed:
                    release = safe_get(c, "release_date")
                    item = {
                        "id": safe_get(c, "id"),
                        "title": safe_get(c, "title"),
                        "year": int(release[:4]) if release else None,
                        "vote_average": safe_get(c, "vote_average"),
                        "overview": safe_get(c, "overview"),
                    }
                    add_candidate(item, 3.5, f"Directed by {dname}")
        except Exception:
            continue

    for rec in candidates.values():
        if rec.tmdb_vote and rec.tmdb_vote >= 7.5:
            rec.score += 0.8
            rec.reasons.append("High TMDB rating")

    # Build natural-language explanations
    for tid, rec in candidates.items():
        seeds = seed_map.get(tid, [])
        rec.explanation = build_explanation(
            title=rec.title,
            reasons=rec.reasons,
            seed_titles=seeds,
            top_genres=top_genre_names,
            top_directors=top_director_names,
            tmdb_vote=rec.tmdb_vote,
        )

    ranked = sorted(candidates.values(), key=lambda r: r.score, reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Letterboxd Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Letterboxd Movie Recommender")
st.markdown(
    "Analyze your Letterboxd ratings and get personalized movie recommendations."
)

with st.sidebar:
    st.header("Settings")
    tmdb_key = st.text_input(
        "TMDB API Key (required)",
        type="password",
        help="Get a free key at themoviedb.org/settings/api",
    )
    top_n = st.slider("Number of recommendations", 5, 30, 12)
    min_rating = st.slider("Min rating for favorites", 3.0, 5.0, 4.0, 0.5)

st.subheader("Your Ratings")
uploaded_file = st.file_uploader(
    "Upload your ratings CSV",
    type=["csv"],
    help="Must have columns: Title, Rating (Year optional)",
)

run_button = st.button("Get Recommendations", type="primary", use_container_width=True)

if run_button:
    if not tmdb_key or not tmdb_key.strip():
        st.error("Please enter your TMDB API key.")
        st.stop()

    if uploaded_file is None:
        st.error("Please upload a CSV file.")
        st.stop()

    with st.spinner("Loading your ratings..."):
        try:
            films = load_from_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV:\n{e}")
            st.stop()

    if len(films) < 3:
        st.error(f"Only found {len(films)} rated films. Need at least 3.")
        st.stop()

    st.success(f"Loaded **{len(films)}** rated films")

    try:
        tmdb = TMDBClient(tmdb_key.strip())
    except Exception as e:
        st.error(f"Failed to connect to TMDB: {e}")
        st.stop()

    films = enrich_films(films, tmdb)

    matched = [f for f in films if f.genres or f.directors]
    if len(matched) == 0:
        st.error("No films could be matched to TMDB. Please check your API key.")
        st.stop()

    with st.spinner("Analyzing your taste..."):
        prefs = analyze_preferences(films, min_rating=min_rating)

    with st.spinner("Generating recommendations..."):
        recs = generate_recommendations(films, prefs, tmdb, top_n=top_n)

    # ---- Results ----
    st.divider()
    st.subheader("Your Taste Profile")

    m1, m2, m3 = st.columns(3)
    m1.metric("Films analyzed", prefs["num_rated"])
    m2.metric(f"Highly rated (≥{min_rating}★)", prefs["num_high"])
    m3.metric("Average rating", f"{prefs['avg_user_rating']:.2f}★")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Favorite Genres**")
        if prefs["top_genres"]:
            for g, s in prefs["top_genres"]:
                st.write(f"• {g}")
        else:
            st.write("_None found_")

        st.markdown("**Favorite Directors**")
        if prefs["top_directors"]:
            for d, s in prefs["top_directors"]:
                st.write(f"• {d}")
        else:
            st.write("_None found_")

    with c2:
        st.markdown("**Favorite Actors**")
        if prefs["top_actors"]:
            for a, s in prefs["top_actors"][:8]:
                st.write(f"• {a}")
        else:
            st.write("_None found_")

    st.divider()
    st.subheader("Recommended Movies")
    st.caption("Score is only used for ranking — higher means a stronger match to your taste.")

    if not recs:
        st.warning("No recommendations could be generated.")
    else:
        for i, r in enumerate(recs, 1):
            year_str = f" ({r.year})" if r.year else ""
            st.markdown(f"### {i}. {r.title}{year_str}")
            st.markdown(r.explanation)
            if r.overview:
                with st.expander("Plot synopsis"):
                    st.write(r.overview)
            st.markdown("---")

st.divider()
st.caption("Upload a CSV with columns Title, Year, Rating")
