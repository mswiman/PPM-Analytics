"""
CTG Analytics - NBA Player Evaluation Platform
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="CTG Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - COMPLETELY NEW DESIGN
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

    /* Global */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Hero Section */
    .hero {
        background: #0a0a0a;
        color: white;
        padding: 80px 60px;
        border-radius: 0;
        margin: -2rem -4rem 3rem -4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(255,107,107,0.1) 0%, rgba(78,205,196,0.1) 100%);
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 72px;
        font-weight: 900;
        margin: 0;
        letter-spacing: -2px;
        position: relative;
    }
    .hero-sub {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 20px;
        color: #888;
        margin-top: 15px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Section Headers */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 42px;
        font-weight: 700;
        color: #1a1a1a;
        margin: 60px 0 30px 0;
        border-bottom: 3px solid #ff6b6b;
        padding-bottom: 15px;
        display: inline-block;
    }

    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #eee;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        transform: translateY(-5px);
    }
    .metric-number {
        font-family: 'Playfair Display', serif;
        font-size: 56px;
        font-weight: 900;
        color: #1a1a1a;
        line-height: 1;
    }
    .metric-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #888;
        margin-top: 10px;
    }

    /* Player Cards - New Style */
    .player-card-v2 {
        background: #fafafa;
        border-left: 5px solid;
        padding: 25px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .player-card-v2:hover {
        background: #fff;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    }
    .player-name {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    .player-meta {
        font-family: 'Source Sans Pro', sans-serif;
        color: #888;
        font-size: 14px;
    }
    .player-stat {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 28px;
        font-weight: 700;
    }

    /* Tier Colors */
    .tier-1 { border-color: #FFD700; }
    .tier-2 { border-color: #C0C0C0; }
    .tier-3 { border-color: #CD853F; }
    .tier-4 { border-color: #4682B4; }
    .tier-5 { border-color: #228B22; }
    .tier-6 { border-color: #808080; }

    /* Methodology Box */
    .method-box {
        background: #f8f9fa;
        border: 2px solid #1a1a1a;
        padding: 40px;
        margin: 30px 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .method-title {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Formula Display */
    .formula {
        background: #1a1a1a;
        color: #4ecdc4;
        padding: 25px;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 16px;
        border-radius: 0;
        margin: 20px 0;
        overflow-x: auto;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 5px 15px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-right: 10px;
    }
    .badge-rookie { background: #ff6b6b; color: white; }
    .badge-prospect { background: #4ecdc4; color: white; }
    .badge-warning { background: #ffeaa7; color: #1a1a1a; }

    /* Navigation */
    .stRadio > div {
        flex-direction: row;
        gap: 0;
    }
    .stRadio > div > label {
        background: #f8f9fa;
        padding: 15px 30px;
        border: 1px solid #eee;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 12px;
    }

    /* Quote */
    .quote {
        font-family: 'Playfair Display', serif;
        font-size: 28px;
        font-style: italic;
        color: #555;
        border-left: 4px solid #ff6b6b;
        padding-left: 30px;
        margin: 40px 0;
    }

    /* Sidebar */
    .css-1d391kg {
        background: #0a0a0a;
    }
    .sidebar .sidebar-content {
        background: #0a0a0a;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data
def load_data():
    data = {}

    if os.path.exists('rapm_rolling_all_years.csv'):
        rapm = pd.read_csv('rapm_rolling_all_years.csv')
        if os.path.exists('pbpstats_data/api/all_players.csv'):
            players = pd.read_csv('pbpstats_data/api/all_players.csv')
            rapm = rapm.merge(players[['player_id', 'player_name']], on='player_id', how='left')
        data['rapm'] = rapm

    if os.path.exists('website_data/player_tiers.csv'):
        data['tiers'] = pd.read_csv('website_data/player_tiers.csv')

    if os.path.exists('website_data/rookies_and_prospects.csv'):
        data['rookies'] = pd.read_csv('website_data/rookies_and_prospects.csv')

    if os.path.exists('website_data/ppm_model_results.csv'):
        data['model'] = pd.read_csv('website_data/ppm_model_results.csv')

    if os.path.exists('website_data/player_career_trajectories.csv'):
        data['trajectories'] = pd.read_csv('website_data/player_career_trajectories.csv')

    if os.path.exists('website_data/yearly_rapm_correlations.csv'):
        data['correlations'] = pd.read_csv('website_data/yearly_rapm_correlations.csv')

    return data

data = load_data()

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px; text-align: center;">
        <h2 style="color: white; font-family: 'Playfair Display', serif; font-size: 28px;">CTG</h2>
        <p style="color: #888; font-size: 12px; letter-spacing: 2px;">ANALYTICS</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Home", "The Study", "RAPM Data", "Player Rankings", "Rookies & Prospects", "Projections"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Load metadata if available
    data_date = "December 2024"
    if os.path.exists('website_data/data_metadata.txt'):
        try:
            with open('website_data/data_metadata.txt', 'r') as f:
                for line in f:
                    if line.startswith('data_as_of:'):
                        data_date = line.split(':')[1].strip()
                        break
        except:
            pass

    st.markdown(f"""
    <div style="color: #666; font-size: 11px; padding: 20px;">
        <p><strong>Data Sources</strong></p>
        <p>PBPStats.com (play-by-play)</p>
        <p>NBA.com Tracking Stats</p>
        <p>Own RAPM Calculations</p>
        <br>
        <p><strong>Data As Of</strong></p>
        <p>{data_date}</p>
        <br>
        <p><strong>RAPM Parameters</strong></p>
        <p>Lambda=150, 3yr rolling</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>CTG Analytics</h1>
        <p class="hero-sub">Advanced NBA Player Evaluation</p>
    </div>
    """, unsafe_allow_html=True)

    # Intro Quote
    st.markdown("""
    <p class="quote">
        "How much does a player's presence on the court actually affect their team's scoring margin?"
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    This platform provides **Regularized Adjusted Plus-Minus (RAPM)** calculations,
    machine learning projections, and player tiering for every NBA player.

    Unlike box score statistics, RAPM isolates individual player impact by accounting for
    teammates and opponents on the court during every possession.
    """)

    # Key Metrics
    st.markdown('<h2 class="section-title">At A Glance</h2>', unsafe_allow_html=True)

    if 'tiers' in data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(data['tiers'])}</div>
                <div class="metric-label">Players Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            mvp = len(data['tiers'][data['tiers']['tier'] == 1])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{mvp}</div>
                <div class="metric-label">MVP Caliber</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            rookies = len(data['tiers'][data['tiers']['player_type'].isin(['rookie_2024', 'prospect_2025'])])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{rookies}</div>
                <div class="metric-label">Rookies & Prospects</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">0.80</div>
                <div class="metric-label">1-Year Model R2</div>
            </div>
            """, unsafe_allow_html=True)

    # Top Players
    st.markdown('<h2 class="section-title">The Best Right Now</h2>', unsafe_allow_html=True)

    if 'tiers' in data:
        top10 = data['tiers'][data['tiers']['player_type'] == 'veteran'].nlargest(10, 'total_rapm')

        for i, (_, p) in enumerate(top10.iterrows(), 1):
            tier = int(p['tier'])
            st.markdown(f"""
            <div class="player-card-v2 tier-{tier}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #888; font-size: 24px; font-weight: 300; margin-right: 15px;">{i}</span>
                        <span class="player-name">{p['name']}</span>
                        <span class="player-meta" style="margin-left: 15px;">{p.get('team_name', '')} | Age {int(p['age']) if pd.notna(p['age']) else 'N/A'}</span>
                    </div>
                    <div style="text-align: right;">
                        <span class="player-stat" style="color: {'#228B22' if p['total_rapm'] > 0 else '#dc3545'};">{p['total_rapm']:+.1f}</span>
                        <span class="player-meta" style="display: block;">Total RAPM</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# THE STUDY PAGE
# =============================================================================
elif page == "The Study":
    st.markdown('<h2 class="section-title">The Methodology</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="method-box">
        <div class="method-title">What is RAPM?</div>
        <p>
        <strong>Regularized Adjusted Plus-Minus</strong> is a regression-based approach to measuring player impact.
        For every possession in every game, we solve a massive system of equations to determine each player's
        individual contribution to their team's scoring margin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The Formula")
    st.markdown("""
    <div class="formula">
    minimize: SUM( (actual_margin - predicted_margin)^2 ) + lambda * SUM( player_coefficients^2 )

    WHERE:
    - actual_margin = points scored - points allowed for each possession
    - predicted_margin = sum of RAPM coefficients for players on court
    - lambda = regularization parameter (we use 150)
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-box">
            <div class="method-title">Key Parameters</div>
            <p><strong>Lambda = 150</strong></p>
            <p>Controls regularization. Validated against Josh's 3-year RAPM from Cleaning The Glass.</p>
            <br>
            <p><strong>3-Year Rolling Window</strong></p>
            <p>Uses 3 seasons of data to stabilize estimates while remaining current.</p>
            <br>
            <p><strong>Minimum 500 Possessions</strong></p>
            <p>Players need sufficient sample to appear in rankings.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-box">
            <div class="method-title">RAPM Priors</div>
            <p><strong>O-RAPM Prior: Secondary Assists</strong></p>
            <p>Correlation: r = 0.33</p>
            <p style="color: #888;">Playmaking that box scores miss</p>
            <br>
            <p><strong>D-RAPM Prior: Rim FG% Allowed</strong></p>
            <p>Correlation: r = 0.24</p>
            <p style="color: #888;">Paint protection ability</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Projection Model (PPM)</h2>', unsafe_allow_html=True)

    st.markdown("""
    To predict future RAPM, we train **Gradient Boosting** models using:
    - Current and career-weighted RAPM
    - Tracking statistics (drives, contests, secondary assists)
    - Quality matchup performance
    - Age curves learned from historical data
    """)

    if 'model' in data:
        st.markdown("### Model Performance")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data['model']['target_years'].astype(str) + ' Year',
            y=data['model']['o_ppm_r2'],
            name='O-PPM',
            marker_color='#4ecdc4'
        ))
        fig.add_trace(go.Bar(
            x=data['model']['target_years'].astype(str) + ' Year',
            y=data['model']['d_ppm_r2'],
            name='D-PPM',
            marker_color='#ff6b6b'
        ))
        fig.update_layout(
            barmode='group',
            height=400,
            yaxis_title='R-squared',
            font=dict(family='Source Sans Pro'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Caveats
    st.markdown('<h2 class="section-title">Important Caveats</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-box" style="border-color: #ff6b6b;">
            <span class="badge badge-warning">CAUTION</span>
            <div class="method-title">Survivorship Bias</div>
            <p>
            Long-term projections (5-7 years) only include players who <em>remained in the league</em>.
            The ~70% who failed, retired, or got injured are excluded from training data.
            </p>
            <p style="margin-top: 15px;">
            <strong>What this means:</strong> We predict well among survivors, not who will survive.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-box" style="border-color: #ffeaa7;">
            <span class="badge badge-warning">NOTE</span>
            <div class="method-title">Lambda Selection</div>
            <p>
            We chose lambda = 150 based on correlation with Josh's RAPM, not on predicting
            actual team wins or game outcomes.
            </p>
            <p style="margin-top: 15px;">
            <strong>Future work:</strong> Validate against team win margins and betting markets.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# RAPM DATA PAGE
# =============================================================================
elif page == "RAPM Data":
    st.markdown('<h2 class="section-title">RAPM by Season</h2>', unsafe_allow_html=True)

    if 'rapm' in data:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            seasons = sorted(data['rapm']['end_season'].unique())
            season = st.selectbox("End Season", seasons, index=len(seasons)-1)

        with col2:
            min_poss = st.slider("Min Possessions", 100, 5000, 500, step=100)

        filtered = data['rapm'][
            (data['rapm']['end_season'] == season) &
            (data['rapm']['possessions'] >= min_poss)
        ].copy()

        st.markdown(f"**{len(filtered)} players** with {min_poss}+ possessions in {season}")

        # Scatter Plot
        fig = px.scatter(
            filtered.head(200),
            x='o_rapm',
            y='d_rapm',
            size='possessions',
            hover_name='player_name',
            color='net_rapm',
            color_continuous_scale='RdYlGn',
            labels={'o_rapm': 'Offensive RAPM', 'd_rapm': 'Defensive RAPM', 'net_rapm': 'Net'}
        )
        fig.update_layout(
            height=500,
            font=dict(family='Source Sans Pro'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#fafafa'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        st.plotly_chart(fig, use_container_width=True)

        # Data Table
        st.markdown("### Full Data")
        display = filtered[['player_name', 'start_season', 'end_season', 'possessions', 'o_rapm', 'd_rapm', 'net_rapm']].copy()
        display.columns = ['Player', 'Start', 'End', 'Poss', 'O-RAPM', 'D-RAPM', 'Net']
        display = display.sort_values('Net', ascending=False)
        st.dataframe(display, hide_index=True, use_container_width=True, height=400)

        # Download
        csv = filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, f"rapm_{season}.csv", "text/csv")

# =============================================================================
# PLAYER RANKINGS PAGE
# =============================================================================
elif page == "Player Rankings":
    st.markdown('<h2 class="section-title">Player Tiers</h2>', unsafe_allow_html=True)

    tier_info = {
        1: ("MVP Caliber", "> +4.0", "#FFD700"),
        2: ("All-NBA", "+2.0 to +4.0", "#C0C0C0"),
        3: ("All-Star", "+1.0 to +2.0", "#CD853F"),
        4: ("Quality Starter", "0.0 to +1.0", "#4682B4"),
        5: ("Rotation", "-1.5 to 0.0", "#228B22"),
        6: ("Bench", "< -1.5", "#808080"),
    }

    # Tier Overview
    cols = st.columns(6)
    for i, (tier, (name, range_, color)) in enumerate(tier_info.items()):
        with cols[i]:
            count = len(data['tiers'][data['tiers']['tier'] == tier]) if 'tiers' in data else 0
            st.markdown(f"""
            <div style="background: {color}20; border-left: 4px solid {color}; padding: 15px; text-align: center;">
                <div style="font-weight: 700; font-size: 12px;">{name}</div>
                <div style="font-size: 10px; color: #888;">{range_}</div>
                <div style="font-size: 28px; font-weight: 900; margin-top: 10px;">{count}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if 'tiers' in data:
        tier_select = st.selectbox(
            "Filter by Tier",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: "All Players" if x == 0 else f"Tier {x}: {tier_info[x][0]}"
        )

        df = data['tiers'][data['tiers']['player_type'] == 'veteran'].copy()
        if tier_select > 0:
            df = df[df['tier'] == tier_select]

        df = df.sort_values('total_rapm', ascending=False)

        for _, p in df.head(30).iterrows():
            tier = int(p['tier'])
            change_3yr = p.get('tier_change_3yr', 0)
            arrow = "Rising" if change_3yr > 0 else ("Falling" if change_3yr < 0 else "Stable")

            st.markdown(f"""
            <div class="player-card-v2 tier-{tier}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="player-name">{p['name']}</span>
                        <span class="player-meta" style="margin-left: 15px;">{p.get('team_name', '')} | Age {int(p['age']) if pd.notna(p['age']) else 'N/A'}</span>
                    </div>
                    <div style="display: flex; gap: 30px; align-items: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #888;">OFF</div>
                            <div style="font-size: 20px; font-weight: 700;">{p['o_rapm']:+.1f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #888;">DEF</div>
                            <div style="font-size: 20px; font-weight: 700;">{p['d_rapm']:+.1f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #888;">TOTAL</div>
                            <div style="font-size: 24px; font-weight: 900; color: {'#228B22' if p['total_rapm'] > 0 else '#dc3545'};">{p['total_rapm']:+.1f}</div>
                        </div>
                        <div style="text-align: center; width: 80px;">
                            <div style="font-size: 14px; color: #888;">3YR TREND</div>
                            <div style="font-size: 14px; color: {'#228B22' if change_3yr > 0 else '#dc3545' if change_3yr < 0 else '#888'};">{arrow}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# ROOKIES & PROSPECTS PAGE
# =============================================================================
elif page == "Rookies & Prospects":
    st.markdown('<h2 class="section-title">Rookies & Prospects</h2>', unsafe_allow_html=True)

    # Methodology Tabs
    meth_tab1, meth_tab2, meth_tab3 = st.tabs(["Overview", "Prior System", "Correlations"])

    with meth_tab1:
        st.markdown("""
        <div class="method-box">
            <div class="method-title">The Rookie Problem</div>
            <p>
            RAPM requires large samples to stabilize (~5000+ possessions). Rookies often have limited minutes
            and no career history. We solve this using <strong>Bayesian shrinkage with informative priors</strong>.
            </p>
            <p style="margin-top: 15px;"><strong>Formula:</strong></p>
            <code style="background: #f5f5f5; padding: 10px; display: block; border-radius: 4px;">
            Adjusted RAPM = (data_weight * raw_RAPM) + (prior_weight * prior)<br>
            where data_weight = possessions / (possessions + 5000)
            </code>
            <p style="margin-top: 15px;">
            For a rookie with 2000 possessions: data_weight = 0.29, prior_weight = 0.71<br>
            For a veteran with 15000 possessions: data_weight = 0.75, prior_weight = 0.25
            </p>
        </div>
        """, unsafe_allow_html=True)

    with meth_tab2:
        st.markdown("""
        <div class="method-box">
            <div class="method-title">Position-Specific Priors</div>
            <p><strong>O-RAPM Prior (all positions):</strong></p>
            <ul>
                <li>Secondary assists per 100 poss (r=0.23)</li>
                <li>Drives per 100 poss (r=0.26)</li>
                <li>Total assists (r=0.32)</li>
            </ul>
            <p style="margin-top: 15px;"><strong>D-RAPM Prior (Guards):</strong></p>
            <ul>
                <li>Steals per 100 poss (r=-0.10) - more steals = better defense</li>
            </ul>
            <p style="margin-top: 15px;"><strong>D-RAPM Prior (Bigs):</strong></p>
            <ul>
                <li>Rim defense FG% allowed (r=0.43) - lower = better defense</li>
                <li>Blocks per 100 poss (r=-0.43) - more blocks = better defense</li>
            </ul>
            <p style="margin-top: 15px;"><strong>Draft Position Prior:</strong></p>
            <ul>
                <li>#1 pick: +1.0 | #3 pick: +0.5 | #7 pick: -0.1 | #17 pick: -0.5 | #24 pick: -0.7</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with meth_tab3:
        st.markdown("### Tracking-to-RAPM Correlations")
        st.markdown("*These correlations validate our prior selection*")

        # Load correlations if available
        if os.path.exists('website_data/tracking_correlations.csv'):
            corr_df = pd.read_csv('website_data/tracking_correlations.csv')
            st.dataframe(corr_df.style.format({'correlation': '{:.3f}'}), use_container_width=True)
        else:
            st.info("Run calculate_tracking_correlations.py to generate correlation data")

        st.markdown("""
        <p style="margin-top: 20px; font-style: italic;">
        Key insight: <strong>Position matters</strong>. For bigs, rim defense FG% has r=0.43 correlation with D-RAPM.
        For guards, this stat is irrelevant. We use steals instead.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if 'tiers' in data:
        # 2024 Rookies
        st.markdown("### 2024 Draft Class")
        rookies_24 = data['tiers'][data['tiers']['player_type'] == 'rookie_2024'].sort_values('draft_position')

        for _, p in rookies_24.iterrows():
            proj_trend = p['proj_3yr'] - p['total_rapm']
            st.markdown(f"""
            <div class="player-card-v2" style="border-color: #ff6b6b;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge badge-rookie">#{int(p['draft_position'])} PICK</span>
                        <span class="player-name" style="margin-left: 10px;">{p['name']}</span>
                        <span class="player-meta" style="margin-left: 15px;">{p.get('team_name', '')} | Age {int(p['age'])}</span>
                    </div>
                    <div style="display: flex; gap: 30px; align-items: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #888;">CURRENT EST</div>
                            <div style="font-size: 22px; font-weight: 700;">{p['total_rapm']:+.1f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #888;">3YR PROJ</div>
                            <div style="font-size: 22px; font-weight: 700; color: #228B22;">{p['proj_3yr']:+.1f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #888;">5YR PROJ</div>
                            <div style="font-size: 22px; font-weight: 700; color: #4682B4;">{p['proj_5yr']:+.1f}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2025 Rookies (drafted 2025, now in NBA)
        rookies_25 = data['tiers'][data['tiers']['player_type'] == 'rookie_2025'].sort_values('draft_position')
        if len(rookies_25) > 0:
            st.markdown("### 2025 Draft Class (In NBA)")
            st.markdown("*These players were just drafted and have limited NBA data*")

            for _, p in rookies_25.iterrows():
                poss = int(p.get('off_poss', 0))
                st.markdown(f"""
                <div class="player-card-v2" style="border-color: #9b59b6;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="badge" style="background: #9b59b6; color: white;">#{int(p['draft_position'])} PICK '25</span>
                            <span class="player-name" style="margin-left: 10px;">{p['name']}</span>
                            <span class="player-meta" style="margin-left: 15px;">{p.get('team_name', '')} | Age {int(p['age'])} | {poss:,} poss</span>
                        </div>
                        <div style="display: flex; gap: 30px; align-items: center;">
                            <div style="text-align: center;">
                                <div style="font-size: 12px; color: #888;">CURRENT</div>
                                <div style="font-size: 22px; font-weight: 700;">{p['total_rapm']:+.1f}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 12px; color: #888;">3YR PROJ</div>
                                <div style="font-size: 22px; font-weight: 700; color: #228B22;">{p['proj_3yr']:+.1f}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 12px; color: #888;">5YR PROJ</div>
                                <div style="font-size: 22px; font-weight: 700; color: #4682B4;">{p['proj_5yr']:+.1f}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 2025 Prospects (not yet in NBA)
        st.markdown("### 2025 Draft Prospects")

        st.markdown("""
        <p class="quote" style="font-size: 20px;">
            These players have not yet played in the NBA. Projections are based entirely on
            expected draft position and historical rookie performance by slot.
        </p>
        """, unsafe_allow_html=True)

        prospects = data['tiers'][data['tiers']['player_type'] == 'prospect_2025'].sort_values('draft_position')

        for _, p in prospects.iterrows():
            st.markdown(f"""
            <div class="player-card-v2" style="border-color: #4ecdc4;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge badge-prospect">PROJ #{int(p['draft_position'])}</span>
                        <span class="player-name" style="margin-left: 10px;">{p['name']}</span>
                        <span class="player-meta" style="margin-left: 15px;">{p.get('team_name', '')} | Age {int(p['age'])}</span>
                    </div>
                    <div style="display: flex; gap: 30px; align-items: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #888;">ROOKIE YR EST</div>
                            <div style="font-size: 22px; font-weight: 700;">{p['total_rapm']:+.1f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #888;">3YR PROJ</div>
                            <div style="font-size: 22px; font-weight: 700; color: #228B22;">{p['proj_3yr']:+.1f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #888;">5YR PROJ</div>
                            <div style="font-size: 22px; font-weight: 700; color: #4682B4;">{p['proj_5yr']:+.1f}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PROJECTIONS PAGE
# =============================================================================
elif page == "Projections":
    st.markdown('<h2 class="section-title">Player Projections</h2>', unsafe_allow_html=True)

    if 'tiers' in data:
        players = sorted(data['tiers']['name'].unique())

        # Default to a star player
        default_idx = players.index('Nikola Jokic') if 'Nikola Jokic' in players else 0
        selected = st.selectbox("Select Player", players, index=default_idx)

        player = data['tiers'][data['tiers']['name'] == selected].iloc[0]

        # Player Header
        tier = int(player['tier'])
        tier_info = {1: "MVP Caliber", 2: "All-NBA", 3: "All-Star", 4: "Quality Starter", 5: "Rotation", 6: "Bench"}

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div style="padding: 30px 0;">
                <h1 style="font-family: 'Playfair Display', serif; font-size: 48px; margin: 0;">{player['name']}</h1>
                <p style="font-size: 18px; color: #888;">{player.get('team_name', '')} | Age {int(player['age']) if pd.notna(player['age']) else 'N/A'} | Tier {tier}: {tier_info[tier]}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align: right; padding: 30px 0;">
                <div style="font-size: 56px; font-weight: 900; font-family: 'Playfair Display', serif; color: {'#228B22' if player['total_rapm'] > 0 else '#dc3545'};">
                    {player['total_rapm']:+.1f}
                </div>
                <div style="font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 2px;">Current RAPM</div>
            </div>
            """, unsafe_allow_html=True)

        # Projection Chart
        st.markdown("### Trajectory & Projection")

        # Build trajectory data
        if 'trajectories' in data:
            history = data['trajectories'][data['trajectories']['name'] == selected].sort_values('season')
        else:
            history = pd.DataFrame()

        fig = go.Figure()

        # Historical
        if len(history) > 0:
            fig.add_trace(go.Scatter(
                x=history['season'],
                y=history['total_rapm'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1a1a1a', width=3),
                marker=dict(size=10)
            ))
            current_yr = history['season'].max()
        else:
            current_yr = 2025

        # Projections
        proj_years = [current_yr + 1, current_yr + 3, current_yr + 5]
        proj_values = [player.get('proj_1yr', player['total_rapm']),
                       player.get('proj_3yr', player['total_rapm']),
                       player.get('proj_5yr', player['total_rapm'])]

        fig.add_trace(go.Scatter(
            x=proj_years,
            y=proj_values,
            mode='lines+markers',
            name='Projection',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=12, symbol='diamond')
        ))

        # Confidence band
        upper = [v + 1.5 for v in proj_values]
        lower = [v - 1.5 for v in proj_values]
        fig.add_trace(go.Scatter(
            x=proj_years + proj_years[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(255,107,107,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band'
        ))

        fig.update_layout(
            height=450,
            xaxis_title='Season',
            yaxis_title='Total RAPM',
            font=dict(family='Source Sans Pro'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#fafafa',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Projection Summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("1-Year", f"{player.get('proj_1yr', 0):+.2f}",
                      f"{player.get('proj_1yr', 0) - player['total_rapm']:+.2f}")
        with col2:
            st.metric("3-Year", f"{player.get('proj_3yr', 0):+.2f}",
                      f"{player.get('proj_3yr', 0) - player['total_rapm']:+.2f}")
        with col3:
            st.metric("5-Year", f"{player.get('proj_5yr', 0):+.2f}",
                      f"{player.get('proj_5yr', 0) - player['total_rapm']:+.2f}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 40px; color: #888;">
    <p style="font-family: 'Playfair Display', serif; font-size: 24px; color: #1a1a1a;">CTG Analytics</p>
    <p>Data: PBPStats, NBA.com, Cleaning The Glass</p>
    <p>Built with Python, Streamlit, Plotly</p>
</div>
""", unsafe_allow_html=True)
