# Vantage Point - VALORANT Scouting Dashboard

> AI-powered competitive VALORANT scouting tool for esports analysts and coaches

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Next.js](https://img.shields.io/badge/next.js-16.1-black)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Data Flow Pipeline](#-data-flow-pipeline)
- [Metrics Generated](#-metrics-generated)
- [Setup Guide](#-setup-guide)
- [API Reference](#-api-reference)
- [Precomputed Mode](#-precomputed-mode)
- [Configuration](#-configuration)
- [Scripts Reference](#️-scripts-reference)
- [Project Structure](#-project-structure-detailed)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

Vantage Point is a comprehensive scouting dashboard that analyzes professional VALORANT match data to generate actionable insights for coaches and analysts. The system pulls data from the GRID Esports Data Platform, processes match events, and uses RAG (Retrieval-Augmented Generation) to produce AI-powered scouting reports.

### Key Capabilities

- **Team Analysis**: Win rates, map performance, site preferences
- **Agent Intelligence**: Composition trends, role distribution, agent pools
- **Player Profiles**: Individual tendencies, signature agents, performance metrics
- **Combat Metrics**: First duel stats, multi-kill patterns, economy decisions
- **AI Insights**: RAG-powered strategic recommendations and counter-strategies

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Next.js)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Landing   │  │  Dashboard  │  │  Tab Views  │  │ Precomputed│ │
│  │   Page      │  │   Page      │  │  Components │  │   Loader   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────┐                             │
│                    │  SSE EventSource │ (Real-time streaming)      │
│                    └─────────────────┘                             │
└────────────────────────────┬───────────────────────────────────────┘
                             │ HTTP/SSE
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   main.py   │  │  analyzer   │  │ rag_engine  │  │ grid_client│ │
│  │  (Routes)   │  │ (Metrics)   │  │ (Insights)  │  │  (Data)    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐                │
│         ▼                    ▼                    ▼                │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Precomputed │  │  Metrics Cache  │  │  Knowledge Base │        │
│  │    JSON      │  │    (JSON)       │  │   (FAISS/Jina)  │        │
│  └─────────────┘  └─────────────────┘  └─────────────────┘        │
└────────────────────────────┬───────────────────────────────────────┘
                             │ HTTPS (GraphQL/REST)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GRID Esports Data Platform                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Central    │  │   Series    │  │   File      │  │   Live     │ │
│  │  Data API   │  │   State     │  │  Download   │  │   Feed     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer           | Technology                         | Purpose                                 |
| --------------- | ---------------------------------- | --------------------------------------- |
| **Frontend**    | Next.js 16.1, React 19, TypeScript | UI, SSE streaming, state management     |
| **Styling**     | Tailwind CSS 4, Framer Motion      | Responsive design, animations           |
| **Backend**     | FastAPI, Uvicorn, Python 3.10+     | API server, async processing            |
| **Analytics**   | Pandas, NumPy                      | Data normalization, metrics computation |
| **AI/ML**       | LangChain, Groq, FAISS, Jina       | RAG pipeline, embeddings, vector search |
| **Data Source** | GRID Esports API                   | Official VALORANT match telemetry       |

---

## ✨ Features

### Dashboard Tabs

| Tab          | Description                                                |
| ------------ | ---------------------------------------------------------- |
| **Overview** | Win rates, site preferences, aggression index, quick stats |
| **Insights** | AI-generated strategic analysis and counter-strategies     |
| **Economy**  | Round-by-round economy patterns, buy habits, eco win rates |
| **Combat**   | First duels, multi-kills, trade efficiency, damage stats   |
| **Maps**     | Per-map win rates, site preferences, compositions          |
| **Agents**   | Agent pool, pick rates, role distribution                  |
| **Players**  | Individual stats, signature agents, tendencies             |
| **Counters** | AI-recommended counter-picks and strategies                |

---

## 📊 Data Flow Pipeline

### 1. Data Fetching (GRID API)

```
GRID GraphQL API → fetch_team_matches() → Match list with metadata
                → fetch_series_events() → Round-by-round kill/plant events
```

### 2. Data Normalization (Analyzer)

```
Raw matches → _normalize_data() → df_rounds (round metadata)
                                → df_players (player stats per match)
```

### 3. Metrics Generation (Two-Phase)

```
Phase 1 (Fast): Basic stats from match metadata (~1-2s)
Phase 2 (Detailed): Event parsing for combat metrics (~5-30s depending on series count)
```

### 4. Insight Generation (RAG)

```
Metrics → Jina Embeddings → FAISS Vector Search → Context Retrieval
       → Groq LLM → Streaming Insights
```

---

## 📈 Metrics Generated

### Macro Metrics (Team-Level)

| Metric                    | Description                                  | Source            |
| ------------------------- | -------------------------------------------- | ----------------- |
| `win_rate`                | Overall match win percentage                 | Match outcomes    |
| `win_rate_by_map`         | Win rate per map (Ascent, Bind, etc.)        | Match metadata    |
| `site_preferences`        | Attack site selection distribution (A/B/C %) | Plant events      |
| `pistol_site_preferences` | First-round site tendencies                  | Round 1/13 events |
| `aggression.style`        | Rush/Default/Slow classification             | Round durations   |
| `aggression.avg_duration` | Average round length (seconds)               | Round timing      |
| `aggression.rush_rate`    | % of rounds under 30s                        | Round timing      |
| `role_distribution`       | Duelist/Initiator/Controller/Sentinel %      | Agent picks       |
| `recent_compositions`     | Most common 5-agent lineups                  | Last N matches    |

### Micro Metrics (Player/Combat-Level)

| Metric                                | Description                 | Source          |
| ------------------------------------- | --------------------------- | --------------- |
| `player_tendencies[].kd_ratio`        | Kill/Death ratio per player | Player stats    |
| `player_tendencies[].first_kill_rate` | Opening duel success %      | Kill events     |
| `player_tendencies[].top_agent`       | Most played agent           | Pick history    |
| `combat_metrics.first_duel_wins`      | Opening kill success rate   | Kill timestamps |
| `combat_metrics.trade_efficiency`     | Successful trade percentage | Kill sequences  |
| `combat_metrics.multi_kills`          | 2k/3k/4k/5k frequencies     | Kill events     |
| `side_metrics.attack_win_rate`        | Attack-side round wins      | Round outcomes  |
| `side_metrics.defense_win_rate`       | Defense-side round wins     | Round outcomes  |
| `economy.avg_loadout_value`           | Average credits spent       | Economy data    |
| `economy.eco_round_win_rate`          | Win % on eco rounds         | Round types     |

### Event-Derived Metrics

| Metric                | Description                        | Computation          |
| --------------------- | ---------------------------------- | -------------------- |
| `first_death_context` | Location/timing of first deaths    | Kill events analysis |
| `ultimate_impact`     | ULT economy and round impact       | Ability events       |
| `man_advantage`       | Win rate in 4v5, 3v5 scenarios     | Player death counts  |
| `discipline`          | Unnecessary peek/rotation deaths   | Position analysis    |
| `site_bias`           | Per-map site execution preferences | Plant coordinates    |
| `pace_metrics`        | Average time-to-execute by site    | Round timestamps     |

---

## 🚀 Setup Guide

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **GRID API Key** (required for live data)
- **Jina AI API Key** (required for embeddings)
- **Groq API Key** (optional, for LLM insights)

### Quick Start

```bash
# 1. Clone and navigate to code directory
cd code

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env  # Edit with your API keys

# 4. Start backend
uvicorn app.main:app --reload --port 8080

# 5. Frontend setup (new terminal)
cd ../frontend
npm install

# 6. Start frontend
npm run dev
```

### Step-by-Step Setup

#### Backend Setup

```bash
cd code/backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Required Dependencies (requirements.txt):**

```
fastapi
uvicorn[standard]
pydantic
pandas
numpy
requests
aiohttp
python-dotenv
langchain
langchain-text-splitters
langchain-groq
groq
sentence-transformers
faiss-cpu
beautifulsoup4
pytest
```

#### Environment Configuration

Create `backend/.env`:

```env
# GRID API (Required for live data)
GRID_API_KEY=your_grid_api_key_here

# Jina AI (Required for embeddings)
JINA_API_KEY=your_jina_api_key_here

# Groq LLM (Optional - for AI insights)
GROQ_API_KEY=your_groq_api_key_here

# Debug mode (uses cached data)
DEBUG_MODE=true

# API Mode: graphql, rest, or auto
GRID_API_MODE=auto

# Include event data (combat metrics)
GRID_INCLUDE_EVENTS=true

# Max series to fetch events for
GRID_EVENTS_MAX_SERIES=12

# Precomputed mode
USE_PRECOMPUTED=false
```

#### Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload --port 8080
```

The API will be available at `http://localhost:8080`

#### Frontend Setup

```bash
cd code/frontend

# Install dependencies
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8080" > .env.local
echo "NEXT_PUBLIC_USE_PRECOMPUTED=false" >> .env.local
```

#### Start Frontend Development Server

```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

---

### ⚡ Speed Up Loading (Optional)

If you don't need live GRID API data, you can significantly speed up dashboard loads by pre-seeding caches:

#### Option 1: Seed Match Cache (Recommended for Development)

Pre-download match data to avoid API calls on each request:

```bash
cd backend

# Seed specific teams
python scripts/seed_match_data.py --teams "Cloud9" "Sentinels" "100 Thieves"

# Or seed ALL available teams (takes longer)
python scripts/seed_all_teams.py
```

Then enable debug mode in `.env`:

```env
DEBUG_MODE=true
```

#### Option 2: Pre-compute Metrics Cache

After seeding matches, pre-compute the metrics to skip analysis on load:

```bash
# Compute metrics for all cached teams
python scripts/precompute_metrics.py

# Or specific teams with custom match limit
python scripts/precompute_metrics.py --teams "Cloud9" --limit 30
```

#### Option 3: Full Precomputed Mode (Frontend-Only)

Generate static JSON files for frontend-only deployment (no backend needed):

```bash
# Start backend first
uvicorn app.main:app --reload --port 8080

# Generate precomputed data (in another terminal)
python scripts/precompute_for_frontend.py --limit 20
```

Then enable precomputed mode in `frontend/.env.local`:

```env
NEXT_PUBLIC_USE_PRECOMPUTED=true
```

**Loading Time Comparison:**

| Mode                     | First Load | Subsequent Loads |
| ------------------------ | ---------- | ---------------- |
| Live API (no cache)      | 30-120s    | 30-120s          |
| Debug mode (match cache) | 5-15s      | 5-15s            |
| Metrics cache            | 2-5s       | 2-5s             |
| Precomputed mode         | <1s        | <1s              |

---

## 📡 API Reference

### Endpoints

#### `GET /api/health`

Health check and configuration status.

**Response:**

```json
{
  "status": "healthy",
  "debug_mode": true,
  "api_mode": "graphql",
  "use_precomputed": false,
  "precomputed_dir": "/path/to/precomputed"
}
```

#### `GET /api/teams`

List all teams with cached match data.

**Response:**

```json
{
  "teams": [
    {
      "name": "Cloud9",
      "slug": "cloud9",
      "match_count": 50,
      "file_size": 245678
    }
  ],
  "count": 16
}
```

#### `POST /api/scout`

Generate a scout report (blocking).

**Request:**

```json
{
  "team_name": "Cloud9",
  "match_limit": 20,
  "map_filter": null,
  "game_title": "VALORANT"
}
```

**Response:**

```json
{
  "team_name": "Cloud9",
  "matches_analyzed": 5,
  "metrics": { ... },
  "insights": { ... }
}
```

#### `GET /api/scout/stream`

Stream scout report generation via SSE.

**Parameters:**

- `team_name` (required): Team name to scout
- `match_limit` (default: 50): Max matches to analyze
- `map_filter` (optional): Filter by specific map
- `game_title` (default: "VALORANT"): Game title

**SSE Event Types:**

```javascript
// Progress update
{ "type": "progress", "stage": "fetch_matches", "progress": 15, "message": "Fetching matches" }

// Warning message
{ "type": "warning", "message": "Loading 12 series may take ~2-3 minutes" }

// Metrics payload (partial)
{ "type": "metrics", "team_name": "Cloud9", "matches_analyzed": 5, "metrics": {...} }

// Streaming insight chunk
{ "type": "insight_chunk", "section": "overview", "content": "Cloud9 demonstrates..." }

// Final report
{ "type": "done", "report": { "team_name": "Cloud9", "metrics": {...}, "insights": {...} } }

// Error
{ "type": "error", "message": "No matches found for team" }
```

#### `GET /api/precomputed/teams`

List teams with precomputed reports.

**Response:**

```json
{
  "teams": [
    {
      "name": "Cloud9",
      "slug": "cloud9",
      "match_count": 5,
      "has_insights": true
    }
  ],
  "count": 16,
  "available": true,
  "generated_at": "2026-02-02T14:52:20Z",
  "match_limit": 20
}
```

#### `GET /api/precomputed/{team_slug}`

Get precomputed report for a specific team.

**Response:** Full scout report JSON

---

## 💾 Precomputed Mode

Precomputed mode allows the frontend to run **without a backend** by serving static JSON files.

### Generate Precomputed Data

```bash
cd backend

# Generate for all teams
python scripts/precompute_for_frontend.py

# Generate for specific teams
python scripts/precompute_for_frontend.py --teams "Cloud9" "Sentinels"

# Custom match limit
python scripts/precompute_for_frontend.py --limit 50

# Custom backend URL
python scripts/precompute_for_frontend.py --backend-url http://localhost:8080
```

**Script Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--teams` | All teams | Specific teams to process |
| `--limit` | 20 | Match limit per team |
| `--backend-url` | http://localhost:8080 | Backend API URL |
| `--output` | frontend/public/precomputed | Output directory |
| `--timeout` | 300 | Request timeout (seconds) |
| `--dry-run` | false | Preview without generating |

### Output Structure

```
frontend/public/precomputed/
├── manifest.json           # Team index with metadata
└── teams/
    ├── cloud9.json         # Full report for Cloud9
    ├── sentinels.json      # Full report for Sentinels
    ├── 100_thieves.json
    └── ...
```

### Enable Precomputed Mode

**Frontend (`frontend/.env.local`):**

```env
NEXT_PUBLIC_USE_PRECOMPUTED=true
NEXT_PUBLIC_PRECOMPUTED_BASE_URL=/precomputed
```

**Backend (`backend/.env`):**

```env
USE_PRECOMPUTED=true
PRECOMPUTED_DIR=/path/to/precomputed
```

### Generation Time Estimates

| Team Count | Match Limit | Approximate Time |
| ---------- | ----------- | ---------------- |
| 1 team     | 20 matches  | ~2-5 seconds     |
| 16 teams   | 20 matches  | ~30-60 seconds   |
| 16 teams   | 50 matches  | ~2-5 minutes     |

**Note:** Times vary based on cached data and API response times.

---

## ⚙️ Configuration

### Backend Environment Variables

| Variable                 | Default            | Description                         |
| ------------------------ | ------------------ | ----------------------------------- |
| `GRID_API_KEY`           | -                  | GRID API authentication key         |
| `DEBUG_MODE`             | false              | Use cached data instead of live API |
| `GRID_FORCE_LIVE`        | false              | Bypass all caches                   |
| `GRID_API_MODE`          | auto               | API mode: graphql, rest, auto       |
| `GRID_INCLUDE_EVENTS`    | true               | Fetch detailed event data           |
| `GRID_EVENTS_MAX_SERIES` | 12                 | Max series to fetch events for      |
| `GRID_SERIES_PAGE_SIZE`  | 50                 | Matches per GraphQL page            |
| `GRID_SERIES_MAX_PAGES`  | 6                  | Max pages to fetch                  |
| `JINA_API_KEY`           | -                  | Jina AI embeddings key              |
| `JINA_EMBEDDING_MODEL`   | jina-embeddings-v3 | Embedding model                     |
| `GROQ_API_KEY`           | -                  | Groq LLM API key                    |
| `RAG_USE_WEB`            | false              | Include web scraping in RAG         |
| `USE_PRECOMPUTED`        | false              | Check precomputed first             |
| `PRECOMPUTED_DIR`        | -                  | Custom precomputed directory        |

### Frontend Environment Variables

| Variable                           | Default               | Description             |
| ---------------------------------- | --------------------- | ----------------------- |
| `NEXT_PUBLIC_API_URL`              | http://localhost:8080 | Backend API URL         |
| `NEXT_PUBLIC_USE_PRECOMPUTED`      | false                 | Enable precomputed mode |
| `NEXT_PUBLIC_PRECOMPUTED_BASE_URL` | /precomputed          | Static files location   |

---

## �️ Scripts Reference

All scripts are located in `backend/scripts/` and should be run from the `backend/` directory.

### precompute_for_frontend.py

**Purpose:** Generate static JSON files for frontend-only deployment by calling the running backend API.

**Requires:** Backend server running at specified URL.

```bash
# Basic usage (all teams, 20 matches each)
python scripts/precompute_for_frontend.py

# Specific teams only
python scripts/precompute_for_frontend.py --teams "Cloud9" "Sentinels" "100 Thieves"

# Custom match limit
python scripts/precompute_for_frontend.py --limit 50

# Custom backend URL
python scripts/precompute_for_frontend.py --backend-url http://localhost:8080

# Preview without generating
python scripts/precompute_for_frontend.py --dry-run

# Custom output directory
python scripts/precompute_for_frontend.py --output /path/to/output

# Extended timeout for large datasets
python scripts/precompute_for_frontend.py --timeout 600
```

| Option          | Default                     | Description                           |
| --------------- | --------------------------- | ------------------------------------- |
| `--teams`       | All available               | Space-separated team names to process |
| `--limit`       | 20                          | Matches per team                      |
| `--backend-url` | http://localhost:8080       | Backend API URL                       |
| `--output`      | frontend/public/precomputed | Output directory                      |
| `--timeout`     | 300                         | Request timeout (seconds)             |
| `--dry-run`     | false                       | Preview mode, no API calls            |

**Output Structure:**

```
frontend/public/precomputed/
├── manifest.json              # Team index with metadata
└── teams/
    ├── cloud9.json            # Full report (~700 lines)
    ├── sentinels.json
    └── ...
```

---

### seed_match_data.py

**Purpose:** Seed the debug cache with match data from GRID API for offline/faster development.

**Requires:** GRID_API_KEY in environment.

```bash
# Seed specific teams
python scripts/seed_match_data.py --teams "Cloud9" "Sentinels"

# Seed with custom match limit
python scripts/seed_match_data.py --teams "Cloud9" --limit 100

# Custom game title
python scripts/seed_match_data.py --teams "Cloud9" --game-title "VALORANT"

# Configure pagination
python scripts/seed_match_data.py --series-page-size 100 --series-max-pages 10

# Retry configuration
python scripts/seed_match_data.py --teams "Cloud9" --max-retries 3 --retry-delay 15
```

| Option               | Default  | Description                           |
| -------------------- | -------- | ------------------------------------- |
| `--teams`            | None     | Team names (comma or space separated) |
| `--limit`            | 50       | Matches to cache per team             |
| `--game-title`       | VALORANT | Game filter                           |
| `--series-page-size` | 50       | GraphQL page size                     |
| `--series-max-pages` | 6        | Max pages to scan                     |
| `--max-retries`      | 0        | Retry count (0 = forever)             |
| `--retry-delay`      | 10       | Base retry delay (seconds)            |

**Output:** Creates `data/debug_cache/{team_slug}_matches.json`

---

### seed_all_teams.py

**Purpose:** Discover and cache match data for ALL teams found in recent GRID series.

**Requires:** GRID_API_KEY in environment.

```bash
# Discover and seed all teams
python scripts/seed_all_teams.py
```

**Behavior:**

1. Fetches 500 recent VALORANT series from GRID
2. Extracts all unique team names
3. Sorts by match count (most active first)
4. Caches up to 50 matches per team
5. Reports success/failure summary

**Output:** Creates multiple `data/debug_cache/{team}_matches.json` files

---

### list_teams.py

**Purpose:** Discover available teams from GRID API and show cached teams.

**Requires:** GRID_API_KEY in environment.

```bash
python scripts/list_teams.py
```

**Output:**

```
Discovering VALORANT teams from recent series...
================================================================================
Found 287 recent series

Discovered 156 unique teams

Top 50 Most Active Teams:
--------------------------------------------------------------------------------
Team Name                                | Matches  | Team ID
--------------------------------------------------------------------------------
Sentinels                                | 45       | 123
Cloud9                                   | 42       | 456
...

Teams with cached match data:
--------------------------------------------------------------------------------
  ✓ Cloud9
  ✓ Sentinels
  ✓ 100 Thieves
```

---

### precompute_metrics.py

**Purpose:** Pre-compute metrics cache for all cached teams to speed up dashboard loads.

**Requires:** Existing match data in `data/debug_cache/`

```bash
# Compute metrics for all cached teams
python scripts/precompute_metrics.py

# Specific teams only
python scripts/precompute_metrics.py --teams "Cloud9" "Sentinels"

# Custom match limit
python scripts/precompute_metrics.py --limit 30

# Force recompute (ignore existing cache)
python scripts/precompute_metrics.py --force

# Preview mode
python scripts/precompute_metrics.py --dry-run
```

| Option         | Default    | Description                     |
| -------------- | ---------- | ------------------------------- |
| `--teams`      | All cached | Specific teams to process       |
| `--limit`      | 20         | Matches for metrics computation |
| `--game-title` | VALORANT   | Game filter                     |
| `--force`      | false      | Recompute even if cache exists  |
| `--dry-run`    | false      | Preview mode                    |

**Output:** Creates `data/debug_cache/metrics_{team}_{limit}_VALORANT_all.json`

---

### probe_grid_api.py

**Purpose:** Debug and test GRID API endpoints with various authentication methods.

**Requires:** GRID_API_KEY in environment.

```bash
# Run all API tests
python scripts/probe_grid_api.py

# Test specific endpoint category
python scripts/probe_grid_api.py --category central-data

# Verbose output
python scripts/probe_grid_api.py --verbose

# Save results to file
python scripts/probe_grid_api.py --output results.json
```

**Tests:**

- GraphQL endpoint connectivity
- Authentication header variants
- Series/match/team queries
- File download endpoints
- Error handling

---

## 📁 Project Structure (Detailed)

```
code/
├── README.md                          # This documentation file
├── instructions.txt                   # Quick start commands for dev
│
├── backend/                           # Python FastAPI backend
│   ├── requirements.txt               # Python dependencies (17 packages)
│   ├── .env                           # Environment configuration (create this)
│   │
│   ├── app/                           # Main application package
│   │   ├── __init__.py               # Package marker
│   │   ├── main.py                   # FastAPI app, routes, SSE streaming (758 lines)
│   │   │                             # - /api/scout (POST) - blocking report
│   │   │                             # - /api/scout/stream (GET) - SSE streaming
│   │   │                             # - /api/teams (GET) - list cached teams
│   │   │                             # - /api/health (GET) - health check
│   │   │                             # - /api/precomputed/* - static data endpoints
│   │   │
│   │   ├── analyzer.py               # ScoutingAnalyzer class (2538 lines)
│   │   │                             # - Data normalization (matches → DataFrames)
│   │   │                             # - Fast metrics (win rate, agents, roles)
│   │   │                             # - Detailed metrics (combat, economy, events)
│   │   │                             # - Event parsing (kills, plants, ultimates)
│   │   │
│   │   ├── models.py                 # Pydantic models (77 lines)
│   │   │                             # - ScoutRequest (team_name, match_limit, etc.)
│   │   │                             # - ScoutReport (metrics, insights)
│   │   │                             # - MetricsSummary (all metric fields)
│   │   │                             # - PlayerTendency, AgentPick, etc.
│   │   │
│   │   ├── settings.py               # Configuration dataclass (98 lines)
│   │   │                             # - Loads all env vars at startup
│   │   │                             # - GRID API settings
│   │   │                             # - RAG settings
│   │   │                             # - Precomputed settings
│   │   │
│   │   ├── grid_client.py            # GRID API client (1181 lines)
│   │   │                             # - GridClient (sync) + AsyncGridClient
│   │   │                             # - GraphQL queries for matches/series
│   │   │                             # - REST fallback endpoints
│   │   │                             # - File download (events, end-state)
│   │   │                             # - Caching layer (debug_cache)
│   │   │
│   │   ├── rag_engine.py             # RAG pipeline (1281 lines)
│   │   │                             # - Jina embeddings integration
│   │   │                             # - FAISS vector index
│   │   │                             # - Knowledge base loading
│   │   │                             # - Groq LLM for insights
│   │   │                             # - Streaming insight generation
│   │   │
│   │   ├── env.py                    # Environment loading utility
│   │   │
│   │   ├── end_state_adapter.py      # GRID end-state format parser
│   │   │
│   │   └── graphql/                  # GraphQL query definitions
│   │       ├── __init__.py
│   │       ├── all_series.graphql    # Query for discovering series
│   │       └── team_matches.graphql  # Query for team match history
│   │                                 # Returns: id, teams, map, segments,
│   │                                 # players, agents, playerStats
│   │
│   ├── data/                         # Data files and caches
│   │   ├── valorant_map_sites.json   # Map site coordinates (130 lines)
│   │   │                             # - Per-map A/B/C site x,y coordinates
│   │   │                             # - Used for plant location analysis
│   │   │                             # - Maps: abyss, ascent, bind, breeze,
│   │   │                             #   corrode, fracture, haven, icebox,
│   │   │                             #   lotus, pearl, split, sunset
│   │   │
│   │   ├── debug_cache/              # Cached API responses
│   │   │   ├── {team}_matches.json   # Match history per team
│   │   │   │                         # - Array of match objects
│   │   │   │                         # - Contains teams, players, segments
│   │   │   │
│   │   │   ├── events_{series_id}.json # Round events per series
│   │   │   │                         # - Kill events with timestamps
│   │   │   │                         # - Plant/defuse events
│   │   │   │                         # - Ability usage
│   │   │   │
│   │   │   └── metrics_{key}.json    # Pre-computed metrics cache
│   │   │                             # - Full metrics summary
│   │   │                             # - Cache key format:
│   │   │                             #   {team}_{limit}_{game}_{map}
│   │   │
│   │   └── knowledge_base/           # RAG knowledge documents
│   │       ├── valorant_domain_knowledge.txt  # Strategy reference (50+ lines)
│   │       │                         # - Round structure & economy states
│   │       │                         # - Role expectations & KPIs
│   │       │                         # - Playstyle taxonomy
│   │       │                         # - Map/site notes
│   │       │                         # - Common micro-errors
│   │       │                         # - Counter-strategy tips
│   │       │
│   │       ├── knowledge_base_docs.md # Auto-generated docs manifest
│   │       │
│   │       ├── .rag_cache/           # FAISS index cache
│   │       │   ├── manifest.json     # Document hashes & chunk ranges
│   │       │   ├── embeddings.npy    # Numpy embedding vectors
│   │       │   └── faiss.index       # FAISS vector index
│   │       │
│   │       └── .insights_cache/      # LLM response cache
│   │           └── {hash}.json       # Cached insight responses
│   │
│   ├── scripts/                      # CLI utilities (see Scripts Reference)
│   │   ├── precompute_for_frontend.py # Generate static JSON (303 lines)
│   │   ├── seed_match_data.py        # Seed cache (273 lines)
│   │   ├── seed_all_teams.py         # Batch seeding (115 lines)
│   │   ├── list_teams.py             # Team discovery (124 lines)
│   │   ├── precompute_metrics.py     # Metrics pre-computation (254 lines)
│   │   └── probe_grid_api.py         # API debugging (656 lines)
│   │
│   └── tests/                        # Test suite
│       └── test_analyzer.py          # Analyzer unit tests
│
├── frontend/                         # Next.js React frontend
│   ├── package.json                  # Node dependencies
│   │                                 # - next 16.1, react 19
│   │                                 # - framer-motion, tailwind-merge
│   │                                 # - react-markdown, rehype-raw
│   │
│   ├── next.config.ts                # Next.js configuration
│   ├── tsconfig.json                 # TypeScript configuration
│   ├── postcss.config.mjs            # PostCSS (Tailwind) config
│   ├── eslint.config.mjs             # ESLint configuration
│   ├── .env.local                    # Local environment overrides
│   │
│   ├── public/                       # Static assets
│   │   └── precomputed/              # Precomputed data (generated)
│   │       ├── manifest.json         # Team index
│   │       │   {
│   │       │     "version": "1.0",
│   │       │     "generated_at": "2026-02-02T...",
│   │       │     "match_limit": 20,
│   │       │     "teams": [
│   │       │       {"name": "Cloud9", "slug": "cloud9", ...}
│   │       │     ]
│   │       │   }
│   │       │
│   │       └── teams/                # Per-team report files
│   │           ├── cloud9.json       # Full scout report (~700 lines)
│   │           ├── sentinels.json
│   │           ├── 100_thieves.json
│   │           └── ...               # 16 teams currently generated
│   │
│   └── src/                          # Source code
│       └── app/                      # Next.js App Router
│           ├── page.tsx              # Landing page (688 lines)
│           │                         # - Team selection dropdown
│           │                         # - Feature cards
│           │                         # - Pipeline visualization
│           │                         # - Precomputed mode detection
│           │
│           ├── layout.tsx            # Root layout (HTML head, fonts)
│           ├── globals.css           # Global styles (Tailwind imports)
│           ├── favicon.ico           # Site favicon
│           │
│           ├── dashboard/            # Dashboard route (/dashboard)
│           │   ├── page.tsx          # Dashboard page (439 lines)
│           │   │                     # - SSE stream handling
│           │   │                     # - Progress state management
│           │   │                     # - Tab navigation
│           │   │                     # - Precomputed data loading
│           │   │
│           │   ├── types.ts          # TypeScript interfaces
│           │   │                     # - ScoutReport, MetricsSummary
│           │   │                     # - StreamProgress, StreamMetrics
│           │   │                     # - ProgressState, etc.
│           │   │
│           │   └── components/       # Dashboard tab components
│           │       ├── index.ts      # Barrel export
│           │       ├── TabNav.tsx    # Tab navigation bar
│           │       ├── StatCards.tsx # Reusable stat display cards
│           │       ├── imageMaps.ts  # Map image URL mappings
│           │       │
│           │       ├── OverviewTab.tsx   # Overview metrics display
│           │       │                     # - Win rate, site bias
│           │       │                     # - Aggression index
│           │       │                     # - Quick stats grid
│           │       │
│           │       ├── InsightsTab.tsx   # AI insights display
│           │       │                     # - Markdown rendering
│           │       │                     # - Section-by-section
│           │       │
│           │       ├── EconomyTab.tsx    # Economy analysis
│           │       │                     # - Buy patterns
│           │       │                     # - Eco win rates
│           │       │
│           │       ├── CombatTab.tsx     # Combat metrics
│           │       │                     # - First duels
│           │       │                     # - Multi-kills
│           │       │                     # - Trade efficiency
│           │       │
│           │       ├── MapsTab.tsx       # Per-map breakdown
│           │       │                     # - Map win rates
│           │       │                     # - Site preferences
│           │       │
│           │       ├── AgentsTab.tsx     # Agent composition
│           │       │                     # - Pick rates
│           │       │                     # - Role distribution
│           │       │
│           │       ├── PlayersTab.tsx    # Player stats
│           │       │                     # - Individual tendencies
│           │       │                     # - Signature agents
│           │       │
│           │       ├── CountersTab.tsx   # Counter-strategies
│           │       │                     # - AI recommendations
│           │       │
│           │       └── AimTrainer.tsx    # Mini aim trainer game
│           │
│           ├── loading/              # Loading states
│           │
│           └── lib/                  # Shared utilities
│               └── utils.ts          # Helper functions (cn, etc.)
│
└── docs/                             # Project documentation
    ├── PRD_Vantage_Point.txt         # Product Requirements Document
    ├── Implementation_Guide_Phases.txt # Development phases
    ├── AI_Prompt_Templates.txt       # LLM prompt templates
    ├── VALORANT_Domain_Knowledge.txt # Game strategy reference
    ├── grid_api_notes.txt            # GRID API documentation
    ├── Planning for AI Assistant Coach.txt # Initial planning
    └── chat-JettRAG.txt              # RAG implementation notes
```

---

## 🔧 Troubleshooting

### Common Issues

#### Backend won't start

```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### "JINA_API_KEY is required"

```bash
# Add to .env file
echo "JINA_API_KEY=your_key_here" >> .env
```

#### "No matches found for team"

- Check team name spelling (case-sensitive)
- Ensure debug cache exists: `ls data/debug_cache/`
- Enable debug mode: `DEBUG_MODE=true`

#### Frontend shows "Loading..." forever

```bash
# Check backend is running
curl http://localhost:8080/api/health

# Check CORS is enabled for frontend port
# Verify NEXT_PUBLIC_API_URL in .env.local
```

#### Precomputed mode not working

```bash
# Check .env.local (takes precedence over .env)
cat frontend/.env.local

# Verify manifest exists
ls frontend/public/precomputed/

# Test manifest directly
curl http://localhost:3000/precomputed/manifest.json
```

### Debug Logging

Enable verbose logging:

```bash
# Backend
DEBUG_MODE=true uvicorn app.main:app --reload --port 8080

# Check timing logs
# [TIMING] fetch_matches: 1.23s (50 matches)
# [TIMING] generate_metrics: 0.45s
```

### Performance Tuning

| Setting                  | Low Resources | High Resources |
| ------------------------ | ------------- | -------------- |
| `GRID_EVENTS_MAX_SERIES` | 4             | 20             |
| `GRID_SERIES_PAGE_SIZE`  | 25            | 100            |
| `match_limit` parameter  | 10            | 50             |

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- [GRID Esports](https://grid.gg) - Official VALORANT data provider
- [Jina AI](https://jina.ai) - Embedding models
- [Groq](https://groq.com) - LLM inference
- [Vercel](https://vercel.com) - Next.js framework

---

_Built with ❤️ for the VALORANT esports community_
