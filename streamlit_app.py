#!/usr/bin/env python3
"""
Letterboxd Movie Recommender — Streamlit Web Version (100% Free Hosting)
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
    tmdb_vote: Optional[float] = None
    overview: Optional[str] = None


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
# Scrape (best-effort)
# ---------------------------------------------------------------------------

def scrape_letterboxd(username: str) -> list[RatedFilm]:
    try:
        from letterboxdpy.user import User
        user = User(username)
    except Exception as e:
        raise RuntimeError(
            f"Could not scrape @{username}.\n\n"
            f"Error: {e}\n\n"
            "Letterboxd often blocks cloud servers.\n"
            "Please use the CSV upload option instead — it's much more reliable."
        )

    films: list[RatedFilm] = []
    try:
        raw = user.get_films()
        items = []
        if isinstance(raw, dict):
            items = raw.get("films") or raw.get("items") or list(raw.values())
        else:
            items = list(raw) if raw else []

        for item in items:
            if not isinstance(item, dict):
                continue
            rating = item.get("rating") or item.get("user_rating") or item.get("stars")
            if rating is None:
                continue
            try:
                rating = float(rating)
            except (TypeError, ValueError):
                continue
            if rating <= 0:
                continue
            title = item.get("name") or item.get("title") or item.get("film", {}).get("name")
            year = item.get("year") or item.get("release_year")
            if title:
                films.append(
                    RatedFilm(
                        title=str(title),
                        year=int(year) if year else None,
                        user_rating=rating,
                    )
                )
    except Exception:
        pass

    seen = set()
    unique = []
    for f in films:
        key = (f.title.lower(), f.year)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


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

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[dict]:
        key = f"search:{title}:{year}"
        if key in self._cache:
            return self._cache[key]
        try:
            results = self.search.movies(title, year=year) if year else self.search.movies(title)
            if results:
                best = results[0]
                data = {
                    "id": best.id,
                    "title": best.title,
                    "year": int(best.release_date[:4]) if getattr(best, "release_date", None) else None,
                    "vote_average": getattr(best, "vote_average", None),
                    "overview": getattr(best, "overview", None),
                }
                self._cache[key] = data
                return data
        except Exception:
            pass
        self._cache[key] = None
        return None

    def get_details(self, tmdb_id: int) -> Optional[dict]:
        key = f"details:{tmdb_id}"
        if key in self._cache:
            return self._cache[key]
        try:
            m = self.movie.details(tmdb_id)
            credits = self.movie.credits(tmdb_id)
            crew = credits.crew if hasattr(credits, "crew") else credits.get("crew", [])
            cast = credits.cast if hasattr(credits, "cast") else credits.get("cast", [])
            directors = []
            for c in crew:
                job = c.get("job") if isinstance(c, dict) else getattr(c, "job", None)
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
                if job == "Director" and name:
                    directors.append(name)
            actors = []
            for c in (cast or [])[:8]:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
                if name:
                    actors.append(name)
            genres = [g["name"] if isinstance(g, dict) else g.name for g in (getattr(m, "genres", None) or [])]
            data = {
                "id": tmdb_id,
                "title": m.title,
                "year": int(m.release_date[:4]) if m.release_date else None,
                "genres": genres,
                "directors": directors,
                "actors": actors,
                "vote_average": getattr(m, "vote_average", None),
                "overview": getattr(m, "overview", None),
            }
            self._cache[key] = data
            return data
        except Exception:
            self._cache[key] = None
            return None

    def get_similar(self, tmdb_id: int, limit: int = 7) -> list[dict]:
        try:
            results = self.movie.similar(tmdb_id)
            return [
                {
                    "id": r.id,
                    "title": r.title,
                    "year": int(r.release_date[:4]) if getattr(r, "release_date", None) else None,
                    "vote_average": getattr(r, "vote_average", None),
                    "overview": getattr(r, "overview", None),
                }
                for r in (results or [])[:limit]
            ]
        except Exception:
            return []

    def get_recommendations(self, tmdb_id: int, limit: int = 5) -> list[dict]:
        try:
            results = self.movie.recommendations(tmdb_id)
            return [
                {
                    "id": r.id,
                    "title": r.title,
                    "year": int(r.release_date[:4]) if getattr(r, "release_date", None) else None,
                    "vote_average": getattr(r, "vote_average", None),
                    "overview": getattr(r, "overview", None),
                }
                for r in (results or [])[:limit]
            ]
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
            results = self.discover.discover_movies(params)
            return [
                {
                    "id": r.id,
                    "title": r.title,
                    "year": int(r.release_date[:4]) if getattr(r, "release_date", None) else None,
                    "vote_average": getattr(r, "vote_average", None),
                    "overview": getattr(r, "overview", None),
                }
                for r in (results or [])[:limit]
            ]
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
# Analysis helpers
# ---------------------------------------------------------------------------

def enrich_films(films: list[RatedFilm], tmdb: TMDBClient, max_enrich: int = 50) -> list[RatedFilm]:
    ranked = sorted(films, key=lambda f: f.user_rating, reverse=True)[:max_enrich]
    progress = st.progress(0, text="Looking up movies on TMDB...")
    for i, film in enumerate(ranked):
        result = tmdb.search_movie(film.title, film.year)
        if result:
            film.tmdb_id = result["id"]
            details = tmdb.get_details(result["id"])
            if details:
                film.genres = details.get("genres", [])
                film.directors = details.get("directors", [])
                film.actors = details.get("actors", [])
                if film.letterboxd_avg is None and details.get("vote_average"):
                    film.letterboxd_avg = round(details["vote_average"] / 2, 2)
        progress.progress((i + 1) / len(ranked), text=f"Looking up {film.title[:40]}...")
        time.sleep(0.04)
    progress.empty()
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


def generate_recommendations(
    films: list[RatedFilm],
    prefs: dict[str, Any],
    tmdb: TMDBClient,
    top_n: int = 15,
) -> list[Recommendation]:
    already_seen = {f.title.lower() for f in films}
    candidates: dict[int, Recommendation] = {}

    def add_candidate(item: dict, base_score: float, reason: str):
        tid = item.get("id")
        title = item.get("title")
        if not tid or not title or title.lower() in already_seen:
            return
        if tid in candidates:
            candidates[tid].score += base_score
            if reason not in candidates[tid].reasons:
                candidates[tid].reasons.append(reason)
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

    high = prefs["high_rated"][:12]

    for f in high:
        if not f.tmdb_id:
            continue
        for sim in tmdb.get_similar(f.tmdb_id, limit=6):
            add_candidate(sim, 3.0 * (f.user_rating / 5), f"Similar to your {f.user_rating}★ {f.title}")
        for rec in tmdb.get_recommendations(f.tmdb_id, limit=4):
            add_candidate(rec, 2.5 * (f.user_rating / 5), f"Recommended alongside {f.title}")

    genre_map = tmdb.genre_name_to_id()
    top_genre_ids = [genre_map[g] for g, _ in prefs["top_genres"][:3] if g in genre_map]
    if top_genre_ids:
        for disco in tmdb.discover_by(top_genre_ids, limit=12):
            add_candidate(
                disco,
                2.0,
                f"Matches preferred genres ({', '.join(g for g, _ in prefs['top_genres'][:3])})",
            )

    for dname, _ in prefs["top_directors"][:4]:
        try:
            person_search = tmdb.search.people(dname)
            if person_search:
                pid = person_search[0].id
                credits = tmdb.person_api.movie_credits(pid)
                crew = credits.crew if hasattr(credits, "crew") else credits.get("crew", [])
                directed = [
                    c for c in crew
                    if (c.get("job") if isinstance(c, dict) else getattr(c, "job", None)) == "Director"
                ][:5]
                for c in directed:
                    if isinstance(c, dict):
                        item = {
                            "id": c.get("id"), "title": c.get("title"),
                            "year": int(c["release_date"][:4]) if c.get("release_date") else None,
                            "vote_average": c.get("vote_average"),
                            "overview": c.get("overview"),
                        }
                    else:
                        item = {
                            "id": c.id, "title": c.title,
                            "year": int(c.release_date[:4]) if getattr(c, "release_date", None) else None,
                            "vote_average": getattr(c, "vote_average", None),
                            "overview": getattr(c, "overview", None),
                        }
                    add_candidate(item, 3.5, f"Directed by {dname}")
        except Exception:
            continue

    for rec in candidates.values():
        if rec.tmdb_vote and rec.tmdb_vote >= 7.5:
            rec.score += 0.8
            rec.reasons.append("High TMDB rating")

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
    "Analyze your Letterboxd ratings and get personalized movie recommendations.\n\n"
    "**How to use:**\n"
    "1. Get a free TMDB API key → [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)\n"
    "2. Upload a CSV of your ratings **(recommended)** or try your Letterboxd username\n"
    "3. Click the button"
)

with st.sidebar:
    st.header("Settings")
    tmdb_key = st.text_input(
        "TMDB API Key (required)",
        type="password",
        help="Get a free key at themoviedb.org/settings/api",
    )
    top_n = st.slider("Number of recommendations", 5, 30, 15)
    min_rating = st.slider("Min rating for favorites", 3.0, 5.0, 4.0, 0.5)

st.subheader("Your Ratings")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload CSV (recommended)",
        type=["csv"],
        help="Must have columns: Title, Rating (Year optional)",
    )

with col2:
    username = st.text_input(
        "Or Letterboxd username",
        placeholder="yourusername",
        help="Often blocked on free cloud servers — CSV is more reliable",
    )

run_button = st.button("Get Recommendations", type="primary", use_container_width=True)

if run_button:
    if not tmdb_key or not tmdb_key.strip():
        st.error("Please enter your TMDB API key.")
        st.stop()

    films: list[RatedFilm] = []

    with st.spinner("Loading your ratings..."):
        if uploaded_file is not None:
            try:
                films = load_from_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV:\n{e}")
                st.stop()
        elif username and username.strip():
            try:
                films = scrape_letterboxd(username.strip())
            except Exception as e:
                st.error(str(e))
                st.stop()
        else:
            st.error("Please upload a CSV or enter a Letterboxd username.")
            st.stop()

    if len(films) < 5:
        st.error(f"Only found {len(films)} rated films. Need at least 5.")
        st.stop()

    st.success(f"Loaded {len(films)} rated films")

    try:
        tmdb = TMDBClient(tmdb_key.strip())
    except Exception as e:
        st.error(f"TMDB connection error: {e}")
        st.stop()

    films = enrich_films(films, tmdb)

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
                st.write(f"• {g} ({s:.1f})")
        else:
            st.write("_Not enough data_")

        st.markdown("**Favorite Directors**")
        if prefs["top_directors"]:
            for d, s in prefs["top_directors"]:
                st.write(f"• {d}")
        else:
            st.write("_Not enough data_")

    with c2:
        st.markdown("**Favorite Actors**")
        if prefs["top_actors"]:
            for a, s in prefs["top_actors"][:8]:
                st.write(f"• {a}")
        else:
            st.write("_Not enough data_")

        if prefs["overrated_by_user"]:
            st.markdown("**Films you love more than the crowd**")
            for f in prefs["overrated_by_user"][:5]:
                st.write(f"• {f.title} ({f.year or '?'}) — you {f.user_rating}★")

    st.divider()
    st.subheader("Recommended Movies")

    if not recs:
        st.info("No recommendations could be generated.")
    else:
        table_data = []
        for i, r in enumerate(recs, 1):
            reasons = "; ".join(r.reasons[:2])
            if len(r.reasons) > 2:
                reasons += f" (+{len(r.reasons)-2} more)"
            table_data.append({
                "#": i,
                "Title": r.title,
                "Year": r.year or "—",
                "Score": f"{r.score:.1f}",
                "Why recommended": reasons,
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "CSV format example: Title,Year,Rating → Parasite,2019,5\n\n"
    "This tool uses public Letterboxd data + TMDB. Not affiliated with Letterboxd."
)
