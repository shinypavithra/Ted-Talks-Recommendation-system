import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="TED Talks Recommender",
    page_icon=None,
    layout="wide",
)


st.markdown("""
<style>
    .stApp { background-color: #121212; }

    body, p, div, span, label { color: #e0e8ef !important; }

    h1 {
        color: #4dd0e1 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 { color: #e0e8ef !important; }

    section[data-testid="stSidebar"] {
        background-color: #0d1f2d !important;
        border-right: 1px solid #1a3a4a !important;
    }
    section[data-testid="stSidebar"] * { color: #e0e8ef !important; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #4dd0e1 !important; }

    .stButton > button {
        background-color: #0d7a8a !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: background-color 0.2s !important;
    }
    .stButton > button:hover { background-color: #09616f !important; }

    .stTextArea textarea {
        background-color: #0d1f2d !important;
        color: #e0e8ef !important;
        border-radius: 10px !important;
        border: 1.5px solid #1a3a4a !important;
        font-size: 15px !important;
    }
    .stTextArea textarea:focus {
        border-color: #4dd0e1 !important;
        box-shadow: 0 0 0 2px rgba(77,208,225,0.15) !important;
    }
    .stTextArea textarea::placeholder { color: #4a7a8a !important; }

    .stSlider > div > div > div > div { background-color: #4dd0e1 !important; }

    [data-testid="stMetricValue"] { color: #4dd0e1 !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #7aabb8 !important; }

    hr { border-color: #1a3a4a !important; }
    .stCaption, small { color: #7aabb8 !important; }

    .talk-card {
        background: #0d1f2d;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border: 1px solid #1a3a4a;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .talk-card:hover {
        border-color: #4dd0e1;
        box-shadow: 0 4px 20px rgba(77,208,225,0.12);
    }

    .rank {
        display: inline-block;
        background: #0d7a8a;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        font-size: 13px;
        margin-right: 8px;
    }

    .tag {
        display: inline-block;
        background: #0a2a35;
        color: #4dd0e1;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px 3px 2px 0;
        border: 1px solid #1a4a5a;
    }

    .score-label {
        font-size: 12px;
        color: #7aabb8;
        margin-bottom: 2px;
    }

    .watch-btn {
        background: #0d7a8a;
        color: white !important;
        padding: 6px 16px;
        border-radius: 8px;
        text-decoration: none !important;
        font-size: 13px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .watch-btn:hover { background: #09616f; }

    .stProgress > div > div > div > div { background-color: #4dd0e1 !important; }
    .stProgress > div > div > div { background-color: #1a3a4a !important; }

    .stSpinner > div { border-top-color: #4dd0e1 !important; }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk()



_stemmer     = PorterStemmer()
_stop_words  = set(stopwords.words('english'))
_punct_table = str.maketrans('', '', string.punctuation)

def preprocess_text(text):
    text  = str(text).lower().translate(_punct_table)
    words = [
        _stemmer.stem(w)
        for w in text.split()
        if w not in _stop_words and len(w) > 2
    ]
    return " ".join(words)



@st.cache_resource
def load_everything():
    # Load dataset
    df = pd.read_csv('ted_main.csv')
    df = df[[
        'main_speaker', 'title', 'description',
        'tags', 'url', 'views', 'speaker_occupation'
    ]].dropna()

    
    df['content'] = (
        df['title'] + ' ' +
        df['description'] + ' ' +
        df['tags']
    ).apply(preprocess_text)


    vectorizer = TfidfVectorizer(analyzer='word')
    vectorizer.fit(df['content'])

   
    all_vectors = vectorizer.transform(df['content'])

    return vectorizer, df, all_vectors


def recommend_talks(user_query, vectorizer, df, all_vectors, n_results):
    cleaned  = preprocess_text(user_query)
    user_vec = vectorizer.transform([cleaned])
    scores   = cosine_similarity(user_vec, all_vectors)[0]
    results  = df.copy()
    results['similarity'] = scores
    return results.sort_values('similarity', ascending=False).head(n_results)


try:
    vectorizer, df, all_vectors = load_everything()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


with st.sidebar:
    st.markdown("## TED Recommender")
    st.markdown("---")
    st.markdown("### Settings")
    n_results = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        value=5
    )

    st.markdown("---")
    
    st.markdown("### Example queries")
    examples = [
        "Climate change and global health",
        "Artificial intelligence and the future",
        "Creativity and innovation in education",
        "Mental health and happiness",
        "Leadership and success at work",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state['query'] = ex



st.markdown("# TED Talks Recommendation System")
st.markdown("Describe a topic you're curious about and get matched to the most relevant TED Talks.")
st.markdown("---")

if not model_loaded:
    st.error(
        f"Could not load ted_main.csv. "
        f"Make sure it is in the same folder as app.py."
        f"\n\nError: {load_error}"
    )
    st.stop()

default_query = st.session_state.get('query', '')
user_input = st.text_area(
    "What would you like to learn about?",
    value=default_query,
    height=100,
    placeholder="e.g. How creativity and curiosity shape the way we learn and grow...",
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    search_clicked = st.button("Find Talks")
with col2:
    if st.button("Clear"):
        st.session_state['query'] = ''
        st.rerun()

st.markdown("---")


if search_clicked:
    if not user_input.strip():
        st.warning("Please enter a topic to search for.")
    else:
        with st.spinner("Finding the best TED Talks for you..."):
            results = recommend_talks(
                user_input, vectorizer, df, all_vectors, n_results
            )

        st.markdown(f"### Top {n_results} talks matching your query")
        st.caption(f'Results for: *"{user_input}"*')
        st.markdown("")

        rank_colors = [
            '#0d7a8a','#0a9aad','#4dd0e1','#0d7a8a','#0a9aad',
            '#0d7a8a','#0a9aad','#4dd0e1','#0d7a8a','#0a9aad',
            '#0d7a8a','#0a9aad','#4dd0e1','#0d7a8a','#0a9aad',
        ]

        for i, (_, row) in enumerate(results.iterrows()):
            similarity_pct = round(row['similarity'] * 100, 1)
            rank_color     = rank_colors[i % len(rank_colors)]

           
            title      = row['title']
            speaker    = row['main_speaker']
            url        = row['url']
            occ        = str(row.get('speaker_occupation', ''))
            occ_html   = (
                f'<span style="color:#7aabb8;font-size:12px"> · {occ}</span>'
                if occ and occ != 'nan' else ''
            )
            desc       = str(row['description'])
            desc_short = desc[:220] + ('...' if len(desc) > 220 else '')
            tag_list   = [t.strip().strip("[]'\" ") for t in str(row['tags']).split(',')[:4]]
            tags_html  = ' '.join([f'<span class="tag">{t}</span>' for t in tag_list if t])

            with st.container():
                st.markdown(f"""
                <div class="talk-card">
                    <div style="display:flex;justify-content:space-between;
                                align-items:flex-start;flex-wrap:wrap;gap:10px">
                        <div style="flex:1;min-width:200px">
                            <span class="rank" style="background:{rank_color}">{i+1}</span>
                            <strong style="font-size:16px;color:#e0e8ef">{title}</strong><br>
                            <span style="color:#7aabb8;font-size:13px;
                                         margin-left:36px">{speaker}</span>
                            {occ_html}
                        </div>
                        <div style="text-align:right;flex-shrink:0">
                            <a href="{url}" target="_blank"
                               class="watch-btn">&#9654; Watch on TED</a>
                        </div>
                    </div>
                    <div style="margin:10px 0 8px;color:#b0ccd6;
                                font-size:13.5px;line-height:1.6">
                        {desc_short}
                    </div>
                    <div style="margin-bottom:10px">{tags_html}</div>
                    <div class="score-label">Match score: {similarity_pct}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(row['similarity']))
                st.markdown("")

        # ── Summary stats ──
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Top match score",    f"{round(results.iloc[0]['similarity']*100,1)}%")
        with col_b:
            st.metric("Talks searched",     f"{len(df):,}")
        with col_c:
            avg_views = int(results['views'].mean()) if 'views' in results.columns else 0
            st.metric("Avg views (results)", f"{avg_views:,}" if avg_views else "N/A")
