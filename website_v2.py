"""
PPM Analytics - NBA Player Evaluation Platform
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="PPM Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container { padding-top: 1rem; max-width: 1200px; }

    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 40px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .header-box h1 { font-size: 42px; margin: 0; font-weight: 700; }
    .header-box p { color: #a0a0a0; margin-top: 10px; }

    .metric-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #4CAF50;
    }
    .metric-box h3 { margin: 0; font-size: 28px; color: #1a1a1a; }
    .metric-box p { margin: 5px 0 0 0; color: #666; font-size: 14px; }

    .method-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 25px;
        margin: 15px 0;
    }
    .method-card h4 { color: #1a1a2e; margin-top: 0; }

    .data-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
    }

    .warning-badge {
        background: #fff3e0;
        color: #e65100;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = {}
    if os.path.exists('rapm_rolling_all_years.csv'):
        data['rapm'] = pd.read_csv('rapm_rolling_all_years.csv')
    if os.path.exists('website_data/player_tiers.csv'):
        data['tiers'] = pd.read_csv('website_data/player_tiers.csv')
    if os.path.exists('website_data/feature_importances.csv'):
        data['features'] = pd.read_csv('website_data/feature_importances.csv')
    if os.path.exists('website_data/tracking_correlations.csv'):
        data['correlations'] = pd.read_csv('website_data/tracking_correlations.csv')
    if os.path.exists('website_data/ppm_model_results.csv'):
        data['model_results'] = pd.read_csv('website_data/ppm_model_results.csv')
    if os.path.exists('website_data/yearly_rapm_correlations.csv'):
        data['yearly_corr'] = pd.read_csv('website_data/yearly_rapm_correlations.csv')
    if os.path.exists('website_data/player_career_trajectories.csv'):
        data['trajectories'] = pd.read_csv('website_data/player_career_trajectories.csv')
    return data

data = load_data()

DATA_DATE = "December 16, 2025"
RAPM_SEASONS = "2023-24 to 2025-26"
LAMBDA = 150

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 15px; text-align: center; background: #1a1a2e; border-radius: 8px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; font-size: 24px;">PPM Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Home", "RAPM Database", "O-PPM Model", "D-PPM Model", "Rookie Priors", "Player Rankings", "Projections"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    **Data As Of:** {DATA_DATE}

    **RAPM Seasons:** {RAPM_SEASONS}

    **Lambda:** {LAMBDA}

    **Source:** PBPStats.com
    """)

if page == "Home":
    st.markdown(f"""
    <div class="header-box">
        <h1>PPM Analytics</h1>
        <p>Player Projection Models for NBA Player Evaluation</p>
        <span class="data-badge">Data as of {DATA_DATE}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## What This Project Does")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>RAPM</h3>
            <p>3-year rolling, λ=150</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>O-PPM / D-PPM</h3>
            <p>Predict future RAPM</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>1-7 Year</h3>
            <p>Projection horizons</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### The Process")
    st.markdown("""
    **1. Calculate RAPM (2012-2026)**
    - Source: Play-by-play stint data from PBPStats.com
    - Method: Ridge regression (λ=150) on 3-year rolling windows
    - Windows: 2012-14, 2013-15, 2014-16, ..., 2023-26
    - Output: O-RAPM and D-RAPM per 100 possessions for each player-window

    **2. Build Tracking Features (2013-2026)**
    - Source: PBPStats.com tracking data (drives, touches, rim defense, etc.)
    - Features: secondary_ast_per_100, potential_ast_per_100, drive_rate, drive_fg_pct,
      rim_dfg, dreb_contest_rate, steals, blocks, o_pts_vs_exp_elite
    - Per-100 possession rates calculated from raw totals

    **3. Train O-PPM**
    - Model: Gradient Boosting (100 trees, max_depth=4)
    - Training: Feature seasons 2012-2022 → Target O-RAPM 2013-2023
    - Testing: Feature seasons 2022-2023 → Target O-RAPM 2023-2024
    - Task: Predict O-RAPM 1-7 years into the future

    **4. Train D-PPM**
    - Model: Gradient Boosting (100 trees, max_depth=4)
    - Training: Feature seasons 2012-2022 → Target D-RAPM 2013-2023
    - Testing: Feature seasons 2022-2023 → Target D-RAPM 2023-2024
    - Task: Predict D-RAPM 1-7 years into the future

    **5. Apply Priors for Low-Sample Players**
    - Applies to: Rookies (1 year), second-year players (2 years), anyone with <5000 poss
    - Why: 3-year rolling RAPM requires ~5000+ possessions to stabilize
    - Method: Bayesian shrinkage toward position-specific tracking priors
    - O-RAPM prior: Based on secondary assists, drives, playmaking
    - D-RAPM prior: Bigs use rim_dfg, Guards use steals

    **6. Validate**
    - Out-of-sample testing on 2023-2024 seasons
    - Metrics: R², Pearson correlation, sample size at each horizon
    """)

    st.markdown("---")

    st.markdown("### Validation Methodology")
    st.markdown("""
    **Training Period:** Feature seasons 2012-2022 → Target RAPM seasons 2013-2023

    **Testing Period:** Feature seasons 2022-2023 → Target RAPM seasons 2023-2024

    The model learns to predict future RAPM using historical data, then is evaluated on
    completely held-out seasons. This simulates real-world usage where we don't know
    the future.

    **O-PPM Results by Horizon:**
    | Horizon | R² | Correlation | N |
    |---------|-----|-------------|-----|
    | 1yr | 0.77 | 0.88 | 1,372 |
    | 2yr | 0.64 | 0.80 | 975 |
    | 3yr | 0.53 | 0.73 | 642 |
    | 5yr | 0.66 | 0.81 | 291 |
    | 5-7yr | 0.75 | 0.87 | 196 |

    **D-PPM Results by Horizon:**
    | Horizon | R² | Correlation | N |
    |---------|-----|-------------|-----|
    | 1yr | 0.73 | 0.86 | 4,642 |
    | 2yr | 0.55 | 0.74 | 3,616 |
    | 3yr | 0.43 | 0.66 | 2,802 |
    | 5yr | 0.53 | 0.73 | 1,791 |
    | 5-7yr | 0.63 | 0.80 | 1,519 |

    **Interpretation:** R² drops at 3-4 years as current RAPM becomes less predictive,
    then recovers at 5-7 years as career-weighted features and tracking skills dominate.
    """)

    st.markdown("---")

    if 'model_results' in data:
        st.markdown("### Model Performance (Out-of-Sample)")

        results = data['model_results']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**O-PPM (Predicting Future O-RAPM)**")
            o_data = results[['target_years', 'o_ppm_r2', 'o_ppm_corr', 'o_ppm_n']].copy()
            o_data.columns = ['Horizon', 'R²', 'Correlation', 'N']
            o_data['R²'] = o_data['R²'].apply(lambda x: f"{x:.3f}")
            o_data['Correlation'] = o_data['Correlation'].apply(lambda x: f"{x:.3f}")
            st.dataframe(o_data, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**D-PPM (Predicting Future D-RAPM)**")
            d_data = results[['target_years', 'd_ppm_r2', 'd_ppm_corr', 'd_ppm_n']].copy()
            d_data.columns = ['Horizon', 'R²', 'Correlation', 'N']
            d_data['R²'] = d_data['R²'].apply(lambda x: f"{x:.3f}")
            d_data['Correlation'] = d_data['Correlation'].apply(lambda x: f"{x:.3f}")
            st.dataframe(d_data, hide_index=True, use_container_width=True)

elif page == "RAPM Database":
    st.markdown(f"""
    <div class="header-box">
        <h1>RAPM Database</h1>
        <p>Regularized Adjusted Plus-Minus | {RAPM_SEASONS} | λ={LAMBDA}</p>
        <span class="data-badge">Data as of {DATA_DATE}</span>
    </div>
    """, unsafe_allow_html=True)

    if 'rapm' in data:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            seasons = sorted(data['rapm']['end_season'].unique(), reverse=True)
            season = st.selectbox("End Season", seasons)

        with col2:
            min_poss = st.slider("Min Possessions", 500, 15000, 5000, step=500)

        st.info("**Note:** RAPM requires ~5,000+ possessions to stabilize. Players below this threshold have high variance and should be interpreted with caution.")

        filtered = data['rapm'][
            (data['rapm']['end_season'] == season) &
            (data['rapm']['possessions'] >= min_poss)
        ].copy()

        st.markdown(f"**{len(filtered)} players** with {min_poss}+ possessions ending in {season}")

        fig = px.scatter(
            filtered,
            x='o_rapm', y='d_rapm',
            size='possessions',
            hover_name='player_name',
            color='net_rapm',
            color_continuous_scale='RdYlGn',
            labels={'o_rapm': 'O-RAPM', 'd_rapm': 'D-RAPM (lower = better)', 'net_rapm': 'Net'}
        )
        fig.update_layout(height=500)
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        st.plotly_chart(fig, use_container_width=True)

        display = filtered[['player_name', 'start_season', 'end_season', 'possessions', 'o_rapm', 'd_rapm', 'net_rapm']].copy()
        display.columns = ['Player', 'Start', 'End', 'Poss', 'O-RAPM', 'D-RAPM', 'Net RAPM']
        display = display.sort_values('Net RAPM', ascending=False)
        display['O-RAPM'] = display['O-RAPM'].apply(lambda x: f"{x:+.2f}")
        display['D-RAPM'] = display['D-RAPM'].apply(lambda x: f"{x:+.2f}")
        display['Net RAPM'] = display['Net RAPM'].apply(lambda x: f"{x:+.2f}")
        st.dataframe(display, hide_index=True, use_container_width=True, height=400)

        csv = filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, f"rapm_{season}.csv", "text/csv")

elif page == "O-PPM Model":
    st.markdown(f"""
    <div class="header-box">
        <h1>O-PPM Model</h1>
        <p>Offensive Player Projection Model - Predicting Future O-RAPM</p>
        <span class="data-badge">Data as of {DATA_DATE}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What O-PPM Does")
    st.markdown("""
    O-PPM predicts a player's **future Offensive RAPM** at various time horizons (1-7 years).
    It uses current O-RAPM plus tracking features that correlate with sustainable offensive impact.
    """)

    st.markdown("---")

    st.markdown("### Training & Testing Methodology")
    st.markdown("""
    **Training Data:**
    - Feature seasons: 2012-2022
    - Target RAPM seasons: 2013-2023
    - ~1,000+ player-season pairs

    **Testing Data (Out-of-Sample):**
    - Feature seasons: 2022-2023
    - Target RAPM seasons: 2023-2024
    - Model never sees test data during training

    **Model:** Gradient Boosting Regressor (100 trees, max_depth=4, learning_rate=0.1)

    **Task:** Given a player's stats in Year N, predict their O-RAPM in Year N+1 (or N+2, N+3, etc.)
    """)

    st.markdown("---")

    st.markdown("### Tracking-to-O-RAPM Correlations")
    st.markdown("*These correlations justify which features we include*")

    if 'correlations' in data:
        o_corr = data['correlations'][data['correlations']['rapm_type'] == 'O-RAPM']
        o_corr = o_corr.sort_values('correlation', ascending=False)

        fig = px.bar(o_corr, x='stat', y='correlation',
                     color='correlation', color_continuous_scale='Greens',
                     labels={'stat': 'Tracking Stat', 'correlation': 'Correlation with O-RAPM'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Feature Importance by Prediction Horizon")

    if 'features' in data:
        horizon = st.selectbox("Select Horizon", ["1yr", "2yr", "3yr", "4yr", "5yr"])

        o_features = data['features'][data['features']['target'] == f'O-PPM {horizon}'].copy()
        o_features = o_features.sort_values('importance', ascending=True)

        fig = px.bar(o_features, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Blues',
                     labels={'importance': 'Importance', 'feature': 'Feature'})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key Insight:** For short-term (1yr), current O-RAPM dominates (76.6%).
        For long-term (3yr+), tracking features like drive rate and playmaking become more important
        as current RAPM's predictive power decays.
        """)

    st.markdown("---")

    st.markdown("### O-PPM Feature Details")
    st.markdown("""
    **Primary Features:**
    | Feature | Description | 1yr Importance |
    |---------|-------------|----------------|
    | o_rapm | Current O-RAPM | 76.6% |
    | potential_ast_per_100 | Passes leading to potential assists | 5.1% |
    | secondary_ast_per_100 | Hockey assists per 100 poss | 4.5% |
    | o_pts_vs_exp_elite | Points vs expected (shot quality) | 4.4% |
    | drive_rate | Drives per possession | 3.9% |
    | drive_fg_pct | FG% on drives | 2.9% |
    | o_rapm_career_wt | Career-weighted O-RAPM | 2.7% |

    **Additional Features (from PBPStats):**
    | Feature | Description |
    |---------|-------------|
    | oreb_per_100 | Offensive rebounds per 100 poss |
    | fg3a_pct | 3-point attempt rate (3PA / FGA) |
    | efg_pct | Effective FG% |
    | ts_pct | True Shooting % |
    | at_rim_frequency | % of shots at the rim |
    | at_rim_accuracy | FG% at the rim |

    **Why These Features?**
    - **Secondary assists** capture playmaking that doesn't show in box score assists
    - **Drive rate + drive FG%** identify players who can create at the rim
    - **Shot quality** (pts vs expected) measures shot selection ability
    - **Offensive rebounds** capture second-chance creation
    - **3-point rate + efficiency** identify modern spacing value
    - **Career-weighted RAPM** provides stability for veterans

    **How Feature Importance Shifts by Horizon:**
    | Feature | 1yr | 3yr | 5-7yr |
    |---------|-----|-----|-------|
    | o_rapm | 76.6% | 32.7% | 17.5% |
    | potential_ast_per_100 | 5.1% | 14.7% | 32.1% |
    | drive_rate | 3.9% | 15.0% | 24.6% |
    | o_rapm_career_wt | 2.7% | 12.8% | 5.3% |

    **Key Insight:** For 1-year predictions, current O-RAPM dominates. For 5-7 year predictions,
    playmaking (potential assists) and driving ability become the strongest predictors because
    these skills are more stable and predictive of long-term offensive impact.
    """)

    st.markdown("---")

    st.markdown("### Model Performance")
    if 'model_results' in data:
        results = data['model_results'][['target_years', 'o_ppm_r2', 'o_ppm_corr', 'o_ppm_n']].copy()
        results.columns = ['Horizon', 'R²', 'Correlation', 'Sample Size']
        results['R²'] = results['R²'].apply(lambda x: f"{x:.3f}")
        results['Correlation'] = results['Correlation'].apply(lambda x: f"{x:.3f}")
        st.dataframe(results, hide_index=True, use_container_width=True)

elif page == "D-PPM Model":
    st.markdown(f"""
    <div class="header-box">
        <h1>D-PPM Model</h1>
        <p>Defensive Player Projection Model - Predicting Future D-RAPM</p>
        <span class="data-badge">Data as of {DATA_DATE}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What D-PPM Does")
    st.markdown("""
    D-PPM predicts a player's **future Defensive RAPM** at various time horizons (1-7 years).
    Note: **Lower D-RAPM = better defense.** Uses current D-RAPM plus tracking features.
    """)

    st.markdown("---")

    st.markdown("### Training & Testing Methodology")
    st.markdown("""
    **Training Data:**
    - Feature seasons: 2012-2022
    - Target RAPM seasons: 2013-2023
    - ~3,000+ player-season pairs

    **Testing Data (Out-of-Sample):**
    - Feature seasons: 2022-2023
    - Target RAPM seasons: 2023-2024
    - Model never sees test data during training

    **Model:** Gradient Boosting Regressor (100 trees, max_depth=4, learning_rate=0.1)

    **Task:** Given a player's stats in Year N, predict their D-RAPM in Year N+1 (or N+2, N+3, etc.)

    **Challenge:** Defense is harder to predict than offense because:
    - More dependent on team scheme and rotations
    - Less measurable from individual stats
    - Rim deterrence (not just blocks) matters
    """)

    st.markdown("---")

    st.markdown("### Tracking-to-D-RAPM Correlations")
    st.markdown("*These correlations justify which features we include*")

    if 'correlations' in data:
        d_corr = data['correlations'][data['correlations']['rapm_type'] == 'D-RAPM']
        d_corr = d_corr.sort_values('correlation', ascending=False)

        fig = px.bar(d_corr, x='stat', y='correlation',
                     color='correlation', color_continuous_scale='Reds',
                     labels={'stat': 'Tracking Stat', 'correlation': 'Correlation with D-RAPM'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - **blocks_per_100 (r=-0.29):** More blocks = lower D-RAPM = better defense
        - **rim_dfg (r=+0.20):** Higher rim FG% allowed = higher D-RAPM = worse defense
        - **steals_per_100 (r=-0.08):** Weak correlation, but position-dependent
        """)

    st.markdown("---")

    st.markdown("### Feature Importance by Prediction Horizon")

    if 'features' in data:
        horizon = st.selectbox("Select Horizon", ["1yr", "2yr", "3yr", "4yr", "5yr"], key="d_horizon")

        d_features = data['features'][data['features']['target'] == f'D-PPM {horizon}'].copy()
        d_features = d_features.sort_values('importance', ascending=True)

        fig = px.bar(d_features, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Oranges',
                     labels={'importance': 'Importance', 'feature': 'Feature'})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key Insight:** Defense is harder to predict than offense.
        For long-term (3yr+), contested rebound rate and career-weighted D-RAPM become dominant
        as current D-RAPM's predictive power decays faster than O-RAPM.
        """)

    st.markdown("---")

    st.markdown("### D-PPM Feature Details")
    st.markdown("""
    **Primary Features:**
    | Feature | Description | 1yr Importance |
    |---------|-------------|----------------|
    | d_rapm | Current D-RAPM | 82.4% |
    | rim_dfg | Rim FG% allowed (lower = better) | 4.3% |
    | steals | Total steals | 4.0% |
    | dreb_contest_rate | Contested defensive rebounds | 3.6% |
    | d_rapm_career_wt | Career-weighted D-RAPM | 3.0% |
    | blocks | Total blocks | 2.6% |

    **Additional Features (from PBPStats):**
    | Feature | Description |
    |---------|-------------|
    | blocks_per_100 | Blocks per 100 defensive poss |
    | deflections | Deflected passes (disruption metric) |
    | def_rim_fga | Rim shots defended (volume) |
    | dreb_per_100 | Defensive rebounds per 100 poss |
    | loose_ball_recoveries | Hustle plays |

    **Why These Features?**
    - **Rim DFG%** captures rim protection better than blocks alone
    - **Blocks** measure shot-blocking ability directly
    - **Deflections** capture perimeter disruption and active hands
    - **Contested rebound rate** shows active defensive effort
    - **Steals** indicate defensive anticipation (position-dependent)
    - **Career-weighted D-RAPM** stabilizes volatile defensive metrics

    **Position-Specific Patterns:**
    - **Bigs:** Rim DFG% and blocks are most predictive
    - **Guards:** Steals and deflections more important for perimeter defense

    **How Feature Importance Shifts by Horizon:**
    | Feature | 1yr | 3yr | 5-7yr |
    |---------|-----|-----|-------|
    | d_rapm | 82.4% | 17.1% | 13.2% |
    | d_rapm_career_wt | 3.0% | 25.0% | 25.3% |
    | dreb_contest_rate | 3.6% | 22.2% | 24.5% |
    | rim_dfg | 4.3% | 13.2% | 11.8% |
    | steals | 4.0% | 14.5% | 13.9% |

    **Key Insight:** D-RAPM importance drops faster than O-RAPM over time. For 5-7 year predictions,
    career-weighted D-RAPM and contested rebound rate become dominant, suggesting defensive
    effort/hustle is more stable than raw defensive impact metrics.
    """)

    st.markdown("---")

    st.markdown("### Model Performance")
    if 'model_results' in data:
        results = data['model_results'][['target_years', 'd_ppm_r2', 'd_ppm_corr', 'd_ppm_n']].copy()
        results.columns = ['Horizon', 'R²', 'Correlation', 'Sample Size']
        results['R²'] = results['R²'].apply(lambda x: f"{x:.3f}")
        results['Correlation'] = results['Correlation'].apply(lambda x: f"{x:.3f}")
        st.dataframe(results, hide_index=True, use_container_width=True)

elif page == "Rookie Priors":
    st.markdown(f"""
    <div class="header-box">
        <h1>Rookie & Low-Sample Priors</h1>
        <p>How we handle players with limited RAPM data</p>
        <span class="warning-badge">High uncertainty for low-sample players</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The Problem")
    st.markdown("""
    Our RAPM uses **3-year rolling windows** (e.g., 2023-24 to 2025-26). This creates issues:

    | Player Type | Seasons in Window | Typical Possessions | Issue |
    |-------------|-------------------|---------------------|-------|
    | Rookies | 1 season | 1,000-2,500 | Only 1/3 of full sample |
    | 2nd-year | 2 seasons | 2,500-5,000 | Only 2/3 of full sample |
    | Veterans | 3 seasons | 5,000-15,000 | Full sample, stable |

    RAPM requires ~5,000+ possessions to stabilize. Rookies and second-year players
    have incomplete windows, making their raw RAPM noisy and unreliable.
    """)

    st.markdown("---")

    st.markdown("### The Solution: Bayesian Shrinkage")
    st.markdown("""
    We blend raw RAPM with **informative priors** based on sample size:

    ```
    Adjusted RAPM = (data_weight × raw_RAPM) + (prior_weight × prior)

    where: data_weight = possessions / (possessions + 5000)
    ```

    **Examples by Player Type:**
    | Player | Possessions | Data Weight | Prior Weight | Effect |
    |--------|-------------|-------------|--------------|--------|
    | Rookie (1500 poss) | 1,500 | 23% | 77% | Heavy shrinkage to prior |
    | 2nd-year (3500 poss) | 3,500 | 41% | 59% | Moderate shrinkage |
    | 3rd-year (6000 poss) | 6,000 | 55% | 45% | Balanced |
    | Veteran (12000 poss) | 12,000 | 71% | 29% | Trust the data |
    """)

    st.markdown("---")

    st.markdown("### Position-Specific Priors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**O-RAPM Prior (All Positions)**")
        st.markdown("""
        - Secondary assists per 100 (r=0.23)
        - Drives per 100 (r=0.25)
        - Total assists (r=0.32)
        """)

    with col2:
        st.markdown("**D-RAPM Prior**")
        st.markdown("""
        **Guards:** Steals per 100 (r=-0.10)
        - More steals = better defense

        **Bigs:** Rim DFG% allowed (r=0.43)
        - Lower rim FG% = better defense
        """)

    st.markdown("---")

    st.markdown("### Draft Position Prior")
    st.markdown("""
    Higher draft picks get a positive prior (historical data shows they perform better):

    | Pick | Prior |
    |------|-------|
    | #1 | +1.0 |
    | #3 | +0.5 |
    | #7 | -0.1 |
    | #17 | -0.5 |
    | #24 | -0.7 |
    """)

    if 'tiers' in data:
        st.markdown("---")
        st.markdown("### Current Rookies")

        rookies = data['tiers'][data['tiers']['player_type'].str.contains('rookie', na=False)]
        if len(rookies) > 0:
            rookies = rookies.sort_values('total_rapm', ascending=False)
            display = rookies[['name', 'team_name', 'draft_position', 'off_poss', 'total_rapm', 'proj_3yr']].copy()
            display.columns = ['Player', 'Team', 'Pick', 'Poss', 'Adj RAPM', '3yr Proj']
            display['Adj RAPM'] = display['Adj RAPM'].apply(lambda x: f"{x:+.2f}")
            display['3yr Proj'] = display['3yr Proj'].apply(lambda x: f"{x:+.2f}")
            st.dataframe(display, hide_index=True, use_container_width=True)

elif page == "Player Rankings":
    st.markdown(f"""
    <div class="header-box">
        <h1>Player Rankings</h1>
        <p>Tiered by Net RAPM (O-RAPM minus D-RAPM)</p>
        <span class="data-badge">Data as of {DATA_DATE}</span>
    </div>
    """, unsafe_allow_html=True)

    tier_info = {
        1: ("MVP Caliber", "> +4.0", "#FFD700"),
        2: ("All-NBA", "+2.0 to +4.0", "#C0C0C0"),
        3: ("All-Star", "+1.0 to +2.0", "#CD853F"),
        4: ("Quality Starter", "0.0 to +1.0", "#4682B4"),
        5: ("Rotation", "-1.5 to 0.0", "#32CD32"),
        6: ("Bench", "< -1.5", "#808080"),
    }

    if 'tiers' in data:
        veterans = data['tiers'][data['tiers']['player_type'] == 'veteran']

        for tier_num in [1, 2, 3, 4, 5, 6]:
            tier_players = veterans[veterans['tier'] == tier_num].sort_values('total_rapm', ascending=False)
            if len(tier_players) > 0:
                name, range_str, color = tier_info[tier_num]
                st.markdown(f"### Tier {tier_num}: {name} ({range_str})")

                display = tier_players[['name', 'team_name', 'o_rapm', 'd_rapm', 'total_rapm']].head(15).copy()
                display.columns = ['Player', 'Team', 'O-RAPM', 'D-RAPM', 'Net RAPM']
                display['O-RAPM'] = display['O-RAPM'].apply(lambda x: f"{x:+.2f}")
                display['D-RAPM'] = display['D-RAPM'].apply(lambda x: f"{x:+.2f}")
                display['Net RAPM'] = display['Net RAPM'].apply(lambda x: f"{x:+.2f}")
                st.dataframe(display, hide_index=True, use_container_width=True)

elif page == "Projections":
    st.markdown(f"""
    <div class="header-box">
        <h1>Player Projections</h1>
        <p>O-PPM and D-PPM combined projections</p>
        <span class="data-badge">Data as of {DATA_DATE}</span>
    </div>
    """, unsafe_allow_html=True)

    if 'tiers' in data:
        players = sorted(data['tiers']['name'].unique())
        default_idx = players.index('Nikola Jokic') if 'Nikola Jokic' in players else 0
        selected = st.selectbox("Select Player", players, index=default_idx)

        player = data['tiers'][data['tiers']['name'] == selected].iloc[0]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"## {player['name']}")
            st.markdown(f"**{player.get('team_name', 'N/A')}** | Age {int(player['age']) if pd.notna(player['age']) else 'N/A'}")

        with col2:
            st.metric("Current Net RAPM", f"{player['total_rapm']:+.2f}")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("1-Year Projection", f"{player.get('proj_1yr', 0):+.2f}",
                      f"{player.get('proj_1yr', 0) - player['total_rapm']:+.2f}")
        with col2:
            st.metric("3-Year Projection", f"{player.get('proj_3yr', 0):+.2f}",
                      f"{player.get('proj_3yr', 0) - player['total_rapm']:+.2f}")
        with col3:
            st.metric("5-Year Projection", f"{player.get('proj_5yr', 0):+.2f}",
                      f"{player.get('proj_5yr', 0) - player['total_rapm']:+.2f}")

        if 'trajectories' in data:
            history = data['trajectories'][data['trajectories']['name'] == selected].sort_values('season')
            if len(history) > 0:
                st.markdown("### Historical RAPM Trajectory")
                fig = px.line(history, x='season', y='net_rapm', markers=True)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: #666;">
    <strong>PPM Analytics</strong> | Data as of {DATA_DATE} | Source: PBPStats.com
</div>
""", unsafe_allow_html=True)
