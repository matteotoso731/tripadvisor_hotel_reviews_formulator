import streamlit as st
from transformers import pipeline

# ============================
# 1) PAGE CONFIG
# ============================

st.set_page_config(
    page_title="TripAdvisor Hotel Review Refiner",
    page_icon="🦉",
    layout="wide",
)

# ============================
# 2) PAGE TITLE (GREEN)
# ============================

st.markdown(
    "<h1 style='color:#00a680; font-weight:700; font-size:2.1rem;'>"
    "TripAdvisor Hotel Review Refiner 🦉"
    "</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="font-size:0.95rem; color:#555; margin-bottom:1.5rem;">'
    'Draft a hotel review and get an AI-refined review with star rating, topics, and a polished version of the text.'
    '</div>',
    unsafe_allow_html=True,
)

# ============================
# 3) STYLE BLOCK
# ============================

st.markdown(
    """
    <style>
    .main { background-color: #f5f7f9; }
    .block-container { padding-top: 2rem; }

    .ta-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border: 1px solid #e3e8ef;
    }

    .ta-stars {
        font-size: 1.8rem;
        color: #00a680;
        line-height: 1.2;
        margin-top: 0.4rem;
    }

    .ta-stars-text {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.2rem;
    }

    .ta-chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background-color: #e5f5f0;
        color: #006f4e;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
        border: 1px solid #b5e1cf;
    }

    .ta-section-title {
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        color: #222;
    }

    .ta-review-box {
        background-color: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e3e8ef;
        padding: 0.9rem 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 0.4rem;
    }

    .ta-meta { font-size: 0.8rem; color: #777; margin-top: 0.4rem; }

    [data-testid="stSidebar"] {
        background-color: #f3f7f5;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# 4) LOAD MODELS (CACHED)
# ============================

@st.cache_resource(show_spinner=True)
def load_rating_pipe():
    return pipeline(
        "text-classification",
        model="LiYuan/amazon-review-sentiment-analysis",
        truncation=True,
    )

@st.cache_resource(show_spinner=True)
def load_aspect_pipe():
    return pipeline(
        "token-classification",
        model="dvquys/ner-finetune-restaurant-reviews-aspects",
        aggregation_strategy="simple",
    )

@st.cache_resource(show_spinner=True)
def load_paraphrase_pipe():
    return pipeline(
        "text2text-generation",
        model="humarin/chatgpt_paraphraser_on_T5_base",
    )

ASPECT_MAP = {
    "FOOD": "Food & Beverage",
    "BEVERAGE": "Food & Beverage",
    "STAFF": "Staff & Service",
    "SERVICE": "Staff & Service",
    "LOCATION": "Location & Ambience",
    "VIEW": "Location & Ambience",
    "AMBIENCE": "Location & Ambience",
}
ALLOWED_TOPICS = set(ASPECT_MAP.values())

# ============================
# 5) FUNCTIONS
# ============================

def predict_stars(review):
    out = load_rating_pipe()(review)[0]
    digits = "".join(ch for ch in out["label"] if ch.isdigit())
    return max(1, min(5, int(digits) if digits else 3))

def extract_aspects(review):
    ents = load_aspect_pipe()(review)
    topics = {ASPECT_MAP.get(e["entity_group"]) for e in ents}
    return sorted(t for t in topics if t in ALLOWED_TOPICS)

def paraphrase(review):
    out = load_paraphrase_pipe()(
        review, max_length=256, num_beams=4, do_sample=False
    )[0]["generated_text"]
    return out.strip()

def stars_to_string(stars):
    return "★" * stars + "☆" * (5 - stars)

# ============================
# 6) SIDEBAR
# ============================

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown(
        """
        This app helps you **optimize hotel reviews** using:
        - ⭐ Sentiment-based star rating  
        - 🧩 Topic / aspect extraction  
        - ✏️ Review paraphrasing  

        Inspired by TripAdvisor and powered by Hugging Face transformers.
        """
    )
    st.divider()
    st.markdown("#### 💡 Tips")
    st.markdown(
        """
        - Write at least **10 words**  
        - Mention **staff, food, ambience, location**  
        - Use it as a tool to **improve, standardize, and correct reviews**
        """
    )

# ============================
# 7) MAIN LAYOUT
# ============================

left, right = st.columns([1.1, 1.1])

# ---------- LEFT SIDE ----------
with left:

    st.markdown("### ✍️ Draft your review")

    hotel_name = st.text_input("Hotel name (optional)", "")

    default_example = (
        "The hotel was in a fantastic location near the city center. The staff were incredibly friendly "
        "and helpful, and the breakfast buffet had a great variety of fresh food. "
        "However, the room was a bit noisy at night."
    )

    use_example = st.checkbox("Use example review to see what the output looks like")

    review_text = st.text_area(
        "Hotel review",
        value=default_example if use_example else "",
        placeholder="Write at least 10 words describing your stay...",
        height=200,
    )

    with st.expander("Optional review metadata"):
        trip_type = st.selectbox(
            "Trip type",
            ["Not specified", "Business", "Couples", "Family", "Friends", "Solo"],
            index=0,
        )
        stay_year = st.selectbox(
            "Year of stay",
            ["Not specified", "2025", "2024", "2023", "2022", "2021"],
            index=0,
        )

    if st.button("🔍 Refine review", use_container_width=True):

        if len(review_text.split()) < 10:
            st.warning("Please write at least 10 words.")
        else:
            st.session_state.output = {
                "review": review_text,
                "hotel": hotel_name,
                "trip": trip_type,
                "year": stay_year,
            }

# ---------- RIGHT SIDE ----------
with right:

    st.markdown("### 📊 Output")

    if "output" not in st.session_state:
        st.info("Refine a review on the left to see the output here.")
    else:
        data = st.session_state.output
        review = data["review"]

        with st.spinner("Generating AI-enhanced review..."):

            stars = predict_stars(review)
            topics = extract_aspects(review)
            refined = paraphrase(review)

        st.markdown('<div class="ta-card">', unsafe_allow_html=True)

        # Header
        title = data["hotel"] if data["hotel"] else "Your stay"
        st.markdown(f"#### 🏨 {title}")

        meta_line = []
        if data["trip"] != "Not specified":
            meta_line.append(f"Trip type: **{data['trip']}**")
        if data["year"] != "Not specified":
            meta_line.append(f"Stayed in: **{data['year']}**")

        if meta_line:
            st.markdown(" · ".join(meta_line))

        # Stars
        st.markdown(
            f"""
            <div class="ta-stars">{stars_to_string(stars)}</div>
            <div class="ta-stars-text">Predicted rating: <b>{stars}/5</b></div>
            """,
            unsafe_allow_html=True,
        )

        # Topics
        st.markdown('<div class="ta-section-title">🧩 Topics detected</div>',
                    unsafe_allow_html=True)

        if topics:
            st.markdown(
                "".join(f'<span class="ta-chip">{t}</span>' for t in topics),
                unsafe_allow_html=True,
            )
        else:
            st.markdown("_No specific topics detected._")

        # Refined review
        st.markdown(
            f'<div class="ta-review-box">{refined}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="ta-meta">AI-generated refined version of your review.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
