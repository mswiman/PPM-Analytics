# NBA Player Analytics Platform
## PhD-Level Interactive Website Architecture

---

## Executive Summary

This document outlines the architecture for a sophisticated, research-grade NBA player analytics platform featuring:
- Real-time RAPM (Regularized Adjusted Plus-Minus) visualizations
- Machine learning-powered player projections (1-7 years)
- Interactive comparison tools
- Publication-quality data exports

---

## 1. Technology Stack

### Frontend
```
Framework:     Next.js 14 (React, Server Components)
Styling:       Tailwind CSS + shadcn/ui components
Charts:        D3.js + Recharts for interactive visualizations
State:         Zustand for client-side state management
Animations:    Framer Motion for smooth transitions
```

### Backend
```
API:           Next.js API Routes (serverless)
Database:      PostgreSQL (Supabase or Neon)
Cache:         Redis (Upstash) for hot data
CDN:           Vercel Edge Network
Auth:          NextAuth.js (optional, for saved comparisons)
```

### Data Pipeline
```
ETL:           Python scripts (daily updates)
Storage:       CSV -> PostgreSQL migration
Updates:       GitHub Actions cron job (nightly)
```

---

## 2. Page Structure

### 2.1 Homepage (`/`)
```
+------------------------------------------------------------------+
|  HEADER: Logo | Search | Leaderboards | Projections | About      |
+------------------------------------------------------------------+
|                                                                    |
|  HERO SECTION                                                      |
|  "NBA Player Analytics: RAPM & Projections"                        |
|  [Search any player...]                                            |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  TOP PLAYERS THIS SEASON (3 cards)                                 |
|  +----------------+  +----------------+  +----------------+        |
|  |  [Photo]       |  |  [Photo]       |  |  [Photo]       |        |
|  |  Player Name   |  |  Player Name   |  |  Player Name   |        |
|  |  O: +4.5       |  |  O: +4.2       |  |  O: +3.9       |        |
|  |  D: -2.1       |  |  D: -1.8       |  |  D: -2.5       |        |
|  |  Total: +6.6   |  |  Total: +6.0   |  |  Total: +6.4   |        |
|  +----------------+  +----------------+  +----------------+        |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  RISING STARS                    |  BIGGEST DECLINES              |
|  (Projected improvers)           |  (Projected decliners)         |
|                                                                    |
+------------------------------------------------------------------+
```

### 2.2 Player Page (`/player/[id]`)
```
+------------------------------------------------------------------+
|  PLAYER HEADER                                                     |
|  +--------+  Nikola Jokic                                          |
|  | PHOTO  |  Denver Nuggets | C | Age 29                           |
|  +--------+  6'11" | 284 lbs | Serbia                              |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  CURRENT RAPM (2024)                                               |
|  +------------------+------------------+------------------+        |
|  |    O-RAPM        |    D-RAPM        |    TOTAL         |        |
|  |    +7.0          |    -0.5          |    +7.5          |        |
|  |    [sparkline]   |    [sparkline]   |    [sparkline]   |        |
|  +------------------+------------------+------------------+        |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  CAREER TRAJECTORY CHART                                           |
|  [Interactive D3 chart showing O-RAPM, D-RAPM over time]           |
|                                                                    |
|       8 |                      *                                   |
|       6 |              *   *       *                               |
|       4 |          *                   *                           |
|       2 |      *                           *                       |
|       0 |--*---------------------------------------                |
|      -2 |                                                          |
|         +--------------------------------------------->            |
|           2017  2018  2019  2020  2021  2022  2023  2024           |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  PROJECTIONS                                                       |
|  +------------+------------+------------+------------+             |
|  |   1 Year   |   3 Year   |   5 Year   |   7 Year   |             |
|  |   +6.8     |   +5.5     |   +4.0     |   +2.5     |             |
|  |   [conf]   |   [conf]   |   [conf]   |   [conf]   |             |
|  +------------+------------+------------+------------+             |
|                                                                    |
|  [Projection uncertainty fan chart]                                |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  TRACKING STATS BREAKDOWN                                          |
|  +---------------+---------------+---------------+                 |
|  | Drives/Game   | Potential Ast | Rim DFG%      |                 |
|  |     8.2       |     12.5      |    58.2%      |                 |
|  +---------------+---------------+---------------+                 |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  SIMILAR PLAYERS (by style & trajectory)                           |
|  [Card] [Card] [Card] [Card]                                       |
|                                                                    |
+------------------------------------------------------------------+
```

### 2.3 Leaderboards (`/leaderboards`)
```
+------------------------------------------------------------------+
|  LEADERBOARD FILTERS                                               |
|  [Season: 2024 v] [Stat: Total RAPM v] [Position: All v]           |
|  [Min Games: 30] [Sort: Descending]                                |
+------------------------------------------------------------------+
|                                                                    |
|  RANK | PLAYER           | TEAM | O-RAPM | D-RAPM | TOTAL | GAMES |
|  -----|------------------|------|--------|--------|-------|-------|
|    1  | Nikola Jokic     | DEN  |  +7.0  |  -0.5  |  +7.5 |   65  |
|    2  | Shai Gilgeous-A. | OKC  |  +5.5  |  -1.8  |  +7.3 |   70  |
|    3  | Luka Doncic      | DAL  |  +6.2  |  -0.2  |  +6.4 |   68  |
|    ...                                                             |
+------------------------------------------------------------------+
|                                                                    |
|  [Export CSV] [Export JSON] [Copy Table]                           |
|                                                                    |
+------------------------------------------------------------------+
```

### 2.4 Compare Players (`/compare`)
```
+------------------------------------------------------------------+
|  PLAYER COMPARISON                                                 |
|  [Search Player 1...] vs [Search Player 2...] [+ Add Player]       |
+------------------------------------------------------------------+
|                                                                    |
|  RADAR CHART (normalized stats)                                    |
|                                                                    |
|                     Scoring                                        |
|                        *                                           |
|           Playmaking     *     Defense                             |
|                   *      |       *                                 |
|                    \     |      /                                  |
|                     \    |     /                                   |
|             Rebounding---+---Efficiency                            |
|                                                                    |
|       -- Player 1    -- Player 2                                   |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  HEAD-TO-HEAD TABLE                                                |
|  +------------------+-----------+-----------+                      |
|  | Metric           | Player 1  | Player 2  |                      |
|  +------------------+-----------+-----------+                      |
|  | O-RAPM           |   +5.2    |   +4.8    |                      |
|  | D-RAPM           |   -1.5    |   -2.1    |                      |
|  | Total RAPM       |   +6.7    |   +6.9    |                      |
|  | 3yr Projection   |   +5.0    |   +5.5    |                      |
|  | ...              |   ...     |   ...     |                      |
|  +------------------+-----------+-----------+                      |
|                                                                    |
+------------------------------------------------------------------+
|                                                                    |
|  CAREER TRAJECTORY OVERLAY                                         |
|  [Line chart with both players' careers overlaid]                  |
|                                                                    |
+------------------------------------------------------------------+
```

### 2.5 Projections Dashboard (`/projections`)
```
+------------------------------------------------------------------+
|  PROJECTION LEADERBOARDS                                           |
|  [Horizon: 3 Year v] [Metric: Total RAPM v] [Age: Under 25 v]      |
+------------------------------------------------------------------+
|                                                                    |
|  RISING STARS (biggest projected improvement)                      |
|  +--------------------------------------------------------------+ |
|  | Player          | Current | 3yr Proj | Change | Confidence   | |
|  |-----------------|---------|----------|--------|--------------|  |
|  | Victor Wemby    |  +2.5   |   +7.0   |  +4.5  | +/- 2.0     |  |
|  | Chet Holmgren   |  +3.0   |   +6.5   |  +3.5  | +/- 1.8     |  |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  DECLINING VETERANS (biggest projected decline)                    |
|  +--------------------------------------------------------------+ |
|  | Player          | Current | 3yr Proj | Change | Confidence   | |
|  |-----------------|---------|----------|--------|--------------|  |
|  | Chris Paul      |  +1.5   |   -1.0   |  -2.5  | +/- 1.5     |  |
|  +--------------------------------------------------------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 3. Interactive Charts (D3.js Specifications)

### 3.1 Career Trajectory Chart
```javascript
// Features:
- Dual y-axis (O-RAPM left, D-RAPM right)
- Hover tooltips with exact values
- Click to compare with league average line
- Zoom/pan for detailed exploration
- Projection cone extending into future years
- Confidence interval shading

// Interactions:
- Hover: Show tooltip with season stats
- Click point: Pin tooltip, show detail panel
- Drag to zoom: Focus on specific period
- Toggle: Show/hide D-RAPM line
```

### 3.2 Projection Uncertainty Fan Chart
```javascript
// Features:
- Central projection line
- 50% confidence interval (darker shade)
- 90% confidence interval (lighter shade)
- Historical data as solid line
- Future projections as dashed line

// Visual:
     Current
        |
   +----|----+
  /     |     \     90% CI
 /   +--|--+   \
/   /   |   \   \   50% CI
---*----*----*----*---- Projection
   2024 2025 2026 2027
```

### 3.3 Player Comparison Radar Chart
```javascript
// Dimensions (normalized 0-100):
- Offensive RAPM percentile
- Defensive RAPM percentile
- Playmaking (assists + secondary assists)
- Rim protection (rim DFG%)
- Self-creation (drive rate)
- Efficiency (TS%)

// Interactions:
- Hover dimension: Show raw values
- Toggle players on/off
- Switch between current and projected
```

### 3.4 Age Curve Visualization
```javascript
// Features:
- X-axis: Age (19-40)
- Y-axis: RAPM
- Show individual player trajectory
- Overlay with league-average age curve
- Position-specific comparison curves
- Highlight peak age range

// Insights:
"This player is tracking above the typical
 curve for their position by +1.2 RAPM"
```

---

## 4. Player Images & Icons

### 4.1 Image Sources
```
Primary:     NBA.com headshots API
             https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png

Fallback:    ESPN headshots
             https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{espn_id}.png

Placeholder: Silhouette by position
             /images/placeholder-{position}.svg
```

### 4.2 Image Implementation
```javascript
// Next.js Image component with fallback
<Image
  src={`https://cdn.nba.com/headshots/nba/latest/260x190/${player.nba_id}.png`}
  fallback={`/images/placeholder-${player.position}.svg`}
  alt={player.name}
  width={260}
  height={190}
  className="rounded-lg object-cover"
/>
```

### 4.3 Team Logos
```
Source:      https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg
Size:        40x40 for inline, 80x80 for headers
```

---

## 5. Data Model (PostgreSQL)

### 5.1 Core Tables
```sql
-- Players
CREATE TABLE players (
  id SERIAL PRIMARY KEY,
  nba_id INTEGER UNIQUE,
  name VARCHAR(100) NOT NULL,
  height_inches INTEGER,
  weight_lbs INTEGER,
  birth_date DATE,
  position VARCHAR(10),
  current_team_id INTEGER REFERENCES teams(id),
  draft_year INTEGER,
  draft_pick INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Seasons (player-season records)
CREATE TABLE player_seasons (
  id SERIAL PRIMARY KEY,
  player_id INTEGER REFERENCES players(id),
  season INTEGER NOT NULL,
  team_id INTEGER REFERENCES teams(id),
  games_played INTEGER,
  minutes FLOAT,

  -- RAPM values
  o_rapm FLOAT,
  d_rapm FLOAT,
  total_rapm FLOAT,

  -- My RAPM (for comparison)
  my_o_rapm FLOAT,
  my_d_rapm FLOAT,
  my_total_rapm FLOAT,

  -- Josh RAPM (ground truth)
  josh_o_rapm FLOAT,
  josh_d_rapm FLOAT,
  josh_total_rapm FLOAT,

  -- Tracking stats
  drives FLOAT,
  drive_rate FLOAT,
  drive_fg_pct FLOAT,
  secondary_ast_per_100 FLOAT,
  potential_ast_per_100 FLOAT,
  rim_dfg FLOAT,
  dreb_contest_rate FLOAT,
  blocks FLOAT,
  steals FLOAT,

  -- Career-weighted
  o_rapm_career_wt FLOAT,
  d_rapm_career_wt FLOAT,

  -- Matchup features
  o_pts_vs_exp_elite FLOAT,
  d_pts_vs_exp_elite FLOAT,

  UNIQUE(player_id, season)
);

-- Projections
CREATE TABLE projections (
  id SERIAL PRIMARY KEY,
  player_id INTEGER REFERENCES players(id),
  base_season INTEGER NOT NULL,
  target_horizon VARCHAR(10) NOT NULL, -- '1yr', '3yr', '5yr', '7yr', '5_7yr'

  o_rapm_projected FLOAT,
  d_rapm_projected FLOAT,
  total_rapm_projected FLOAT,

  -- Confidence intervals
  o_rapm_ci_low FLOAT,
  o_rapm_ci_high FLOAT,
  d_rapm_ci_low FLOAT,
  d_rapm_ci_high FLOAT,

  confidence_level FLOAT, -- 0-1 based on data quality

  UNIQUE(player_id, base_season, target_horizon)
);

-- Model performance (for transparency)
CREATE TABLE model_metrics (
  id SERIAL PRIMARY KEY,
  target_horizon VARCHAR(10),
  o_ppm_r2 FLOAT,
  o_ppm_corr FLOAT,
  d_ppm_r2 FLOAT,
  d_ppm_corr FLOAT,
  n_samples INTEGER,
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### 5.2 Indexes for Performance
```sql
CREATE INDEX idx_player_seasons_season ON player_seasons(season);
CREATE INDEX idx_player_seasons_total_rapm ON player_seasons(total_rapm DESC);
CREATE INDEX idx_projections_player ON projections(player_id);
CREATE INDEX idx_players_name ON players(name);
```

---

## 6. API Endpoints

### 6.1 REST API
```
GET  /api/players                    # List all players (paginated)
GET  /api/players/:id                # Single player detail
GET  /api/players/:id/seasons        # Player's season-by-season data
GET  /api/players/:id/projections    # Player's projections
GET  /api/players/search?q=          # Search players by name

GET  /api/leaderboards               # Current season leaderboard
GET  /api/leaderboards/:season       # Historical leaderboard

GET  /api/projections                # Projection leaderboards
GET  /api/projections/rising         # Biggest projected improvers
GET  /api/projections/declining      # Biggest projected decliners

GET  /api/compare?ids=1,2,3          # Compare multiple players

GET  /api/model/metrics              # Model performance stats
```

### 6.2 Response Format
```json
{
  "player": {
    "id": 1,
    "nba_id": 203999,
    "name": "Nikola Jokic",
    "height": "6-11",
    "weight": 284,
    "age": 29,
    "position": "C",
    "team": {
      "id": 7,
      "name": "Denver Nuggets",
      "abbreviation": "DEN"
    }
  },
  "current_season": {
    "season": 2024,
    "o_rapm": 7.0,
    "d_rapm": -0.5,
    "total_rapm": 7.5,
    "games": 65,
    "tracking": {
      "drives": 8.2,
      "secondary_ast": 1.8,
      "rim_dfg": 0.58
    }
  },
  "career": [
    {"season": 2017, "o_rapm": 2.1, "d_rapm": 0.5, "total_rapm": 1.6},
    // ...
  ],
  "projections": {
    "1yr": {"o_rapm": 6.8, "d_rapm": -0.3, "total_rapm": 7.1, "ci": 1.5},
    "3yr": {"o_rapm": 5.5, "d_rapm": 0.2, "total_rapm": 5.3, "ci": 2.0},
    "5yr": {"o_rapm": 4.0, "d_rapm": 0.5, "total_rapm": 3.5, "ci": 2.5},
    "7yr": {"o_rapm": 2.5, "d_rapm": 0.8, "total_rapm": 1.7, "ci": 3.0}
  }
}
```

---

## 7. Performance Optimizations

### 7.1 Caching Strategy
```
Level 1: Browser cache (stale-while-revalidate)
         - Player data: 1 hour
         - Leaderboards: 15 minutes
         - Projections: 24 hours

Level 2: Edge cache (Vercel/Cloudflare)
         - Static pages: 1 hour
         - API responses: 5 minutes

Level 3: Redis cache (hot data)
         - Top 100 player lookups
         - Search autocomplete data
         - Leaderboard pre-computation
```

### 7.2 Data Loading
```javascript
// Incremental Static Regeneration (ISR)
export async function getStaticProps() {
  const player = await getPlayer(id);
  return {
    props: { player },
    revalidate: 3600, // 1 hour
  };
}

// Server Components for fresh data
async function PlayerRAPM({ playerId }) {
  const data = await fetch(`/api/players/${playerId}`, {
    next: { revalidate: 300 } // 5 minutes
  });
  return <RAPMDisplay data={data} />;
}
```

---

## 8. Export Capabilities

### 8.1 Data Export Formats
```
CSV:    Standard comma-separated (Excel compatible)
JSON:   API response format
PNG:    Chart screenshots (html2canvas)
PDF:    Full player reports (react-pdf)
```

### 8.2 Academic Export
```bibtex
@misc{ctg_rapm_2024,
  title = {NBA Player RAPM Analytics},
  author = {CTG Analytics},
  year = {2024},
  url = {https://your-site.com/player/203999},
  note = {Accessed: 2024-12-15}
}
```

---

## 9. Accessibility & SEO

### 9.1 Accessibility (WCAG 2.1 AA)
```
- Alt text for all player images
- Color contrast ratios > 4.5:1
- Keyboard navigation for all interactions
- Screen reader announcements for chart data
- Focus indicators on interactive elements
```

### 9.2 SEO Optimization
```
- Dynamic meta tags per player
- Structured data (JSON-LD) for players
- Sitemap with all player pages
- Canonical URLs
- Open Graph images for social sharing
```

---

## 10. Deployment Architecture

```
                    +------------------+
                    |   Vercel Edge    |
                    |   (CDN/Cache)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+         +----------v---------+
     |   Next.js App   |         |   Vercel Postgres  |
     |   (Serverless)  |         |   (or Supabase)    |
     +--------+--------+         +----------+---------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v--------+
                   |   Redis Cache   |
                   |   (Upstash)     |
                   +-----------------+
```

---

## 11. Development Roadmap

### Phase 1: MVP (Week 1-2)
- [x] Data pipeline (CSV -> PostgreSQL)
- [ ] Basic player pages
- [ ] Simple leaderboards
- [ ] Deployment to Vercel

### Phase 2: Visualization (Week 3-4)
- [ ] D3 career trajectory charts
- [ ] Projection fan charts
- [ ] Player comparison tool

### Phase 3: Polish (Week 5-6)
- [ ] Mobile responsive design
- [ ] Search autocomplete
- [ ] Export functionality
- [ ] Performance optimization

### Phase 4: Advanced (Week 7+)
- [ ] User accounts (saved comparisons)
- [ ] Email alerts (player updates)
- [ ] API access for researchers
- [ ] Embeddable widgets

---

## 12. CSV Files Required

All files are in `website_data/` directory:

| File | Description | Rows |
|------|-------------|------|
| master_player_features.csv | Complete feature set | 6,023 |
| rapm_comparison.csv | My vs Josh RAPM | 6,023 |
| current_season_players.csv | 2024 data | 514 |
| player_career_trajectories.csv | Career history | - |
| ppm_model_results.csv | Model metrics | 7 |
| feature_importances.csv | Feature weights | - |
| yearly_rapm_correlations.csv | Year-by-year validation | 11 |
| top_players_by_season.csv | Leaderboards | - |
| rapm_priors.csv | Prior features | - |
| shot_quality_features.csv | Shot tracking | - |
| o_ppm_elite_matchups.csv | Offensive matchups | 1,471 |
| d_ppm_elite_matchups.csv | Defensive matchups | 2,010 |

---

## 13. Getting Started

```bash
# 1. Clone repository
git clone https://github.com/your-repo/nba-analytics.git
cd nba-analytics

# 2. Install dependencies
npm install

# 3. Set up environment variables
cp .env.example .env.local
# Add DATABASE_URL, REDIS_URL, etc.

# 4. Import data
npm run db:migrate
npm run data:import

# 5. Start development server
npm run dev
```

---

## Contact

For questions about methodology or data access:
- Documentation: `/about` page
- API docs: `/docs/api`
- Research inquiries: research@your-site.com
