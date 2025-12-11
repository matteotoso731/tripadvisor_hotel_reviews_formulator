import streamlit as st
from transformers import pipeline

# ============================
# 1) PAGE CONFIG & STYLE
# ============================

st.set_page_config(
    page_title="TripAdvisor Hotel Review Refiner",
    page_icon="🦉",
    layout="wide",
)

st.markdown(
    """
    <style>
    h1 {
        color: #00a680 !important; /* TripAdvisor green */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Light TripAdvisor-ish theming
st.markdown(
    """
    <style>
    /* Global */
    .main {
        background-color: #f5f7f9;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Subtitle under title */
    .ta-subtitle {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.5rem;
    }

    /* Card containers */
    .ta-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border: 1px solid #e3e8ef;
    }

    /* Star rating */
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

    /* Topic chips */
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

    /* Section titles inside output block */
    .ta-section-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.4rem;
        color: #222;
        margin-top: 1rem;
    }

    /* Cleaned review text */
    .ta-review-box {
        background-color: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e3e8ef;
        padding: 0.8rem 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 0.4rem;
    }

    /* Small meta text */
    .ta-meta {
        font-size: 0.8rem;
        color: #777;
        margin-top: 0.4rem;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #f3f7f5;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# 2) PIPELINES (CACHED)
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
    ASPECT_MODEL_NAME = "dvquys/ner-finetune-restaurant-reviews-aspects"
    return pipeline(
        "token-classification",
        model=ASPECT_MODEL_NAME,
        aggregation_strategy="simple",
    )

@st.cache_resource(show_spinner=True)
def load_paraphrase_pipe():
    PARA_MODEL_ID = "humarin/chatgpt_paraphraser_on_T5_base"
    return pipeline(
        "text2text-generation",
        model=PARA_MODEL_ID,
    )


# Map raw model labels to business-level topics
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
# 3) HELPER FUNCTIONS
# ============================

def predict_stars(review: str) -> int:
    rating_pipe = load_rating_pipe()
    out = rating_pipe(review)[0]
    label = out["label"]
    digits = "".join(ch for ch in label if ch.isdigit())
    if digits:
        stars = int(digits)
    else:
        stars = 3
    return max(1, min(5, stars))


def extract_aspects(review: str):
    aspect_pipe = load_aspect_pipe()
    ents = aspect_pipe(review)
    topics = set()
    for e in ents:
        raw_label = e.get("entity_group")
        mapped_topic = ASPECT_MAP.get(raw_label)
        if mapped_topic in ALLOWED_TOPICS:
            topics.add(mapped_topic)
    return sorted(topics)


def paraphrase_review(review: str) -> str:
    paraphrase_pipe = load_paraphrase_pipe()
    out = paraphrase_pipe(
        review,
        max_length=256,
        num_beams=4,
        do_sample=False,
    )[0]["generated_text"]
    return out.strip()


def stars_to_string(stars: int) -> str:
    stars = max(1, min(5, stars))
    return "★" * stars + "☆" * (5 - stars)


# ============================
# 4) SIDEBAR
# ============================

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown(
        """
        This app helps you **optimize hotel reviews** using:
        - ⭐ Sentiment-based star rating  
        - 🧩 Topic / aspect extraction  
        - ✏️ Review paraphrasing  

        Inspired by a **TripAdvisor-style** review experience and built
        as a deep-learning business application using Hugging Face pipelines.
        """
    )
    st.divider()
    st.markdown("#### 💡 Tips")
    st.markdown(
        """
        - Write at least **10 words**  
        - Describe **food, staff, location, ambience**  
        - Use it as a tool for: **improve, standardize and correct reviews**
        """
    )


# ============================
# 5) MAIN LAYOUT
# ============================

st.title("TripAdvisor-Style Hotel Review Refiner 🦉")

st.markdown(
    '<div class="ta-subtitle">Draft a hotel review and get an AI-refined review with star rating, topics, and a polished version of the text.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 1.1])

# ---------- LEFT: INPUT ----------
with left_col:
    st.markdown("### ✍️ Draft your review")

    # Hotel name (optional)
    hotel_name = st.text_input(
        "Hotel name (optional)",
        placeholder="e.g. Grand Ocean View Hotel",
    )

    default_example = (
        "The hotel was in a fantastic location near the city center. The staff were incredibly friendly "
        "and helpful, and the breakfast buffet had a great variety of fresh food. "
        "However, the room was a bit noisy at night."
    )

    # Checkbox: if ticked, we show the example directly in the textarea
    use_example = st.checkbox(
        "Use example review to see what the output looks like",
        value=False,
    )

    if use_example:
        review_text = st.text_area(
            "Hotel review",
            value=default_example,
            height=200,
        )
    else:
        review_text = st.text_area(
            "Hotel review",
            placeholder="Write at least 10 words describing your stay, staff, food, location, ambience...",
            height=200,
        )

    # Trip type + year grouped in a single expander (as optional metadata)
    with st.expander("Optional review metadata", expanded=False):
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

    # Button with magnifying glass
    refine_button = st.button("🔍 Refine review", use_container_width=True)

    if refine_button:
        if not review_text.strip():
            st.warning("Review text cannot be empty.")
        else:
            word_count = len(review_text.split())
            if word_count < 10:
                st.warning("Please enter a longer review (at least 10 words).")
            else:
                st.session_state["last_review"] = review_text
                st.session_state["last_meta"] = {
                    "hotel_name": hotel_name,
                    "trip_type": trip_type,
                    "stay_year": stay_year,
                }

# ---------- RIGHT: OUTPUT ----------
with right_col:
    st.markdown("### 📊 Output")

    if "last_review" not in st.session_state:
        st.info("Refine a review on the left to see the output here.")
    else:
        review = st.session_state["last_review"]
        meta = st.session_state.get("last_meta", {})

        with st.spinner("Running Hugging Face pipelines on your review..."):
            stars = predict_stars(review)
            star_string = stars_to_string(stars)
            topics = extract_aspects(review)
            cleaned_review = paraphrase_review(review)

        # Single big block that contains all the output
        st.markdown('<div class="ta-card">', unsafe_allow_html=True)

        # Header with hotel name + meta
        title_line = meta.get("hotel_name") or "Your stay"
        st.markdown(f"#### 🏨 {title_line}")

        meta_bits = []
        if meta.get("trip_type") and meta["trip_type"] != "Not specified":
            meta_bits.append(f"Trip type: **{meta['trip_type']}**")
        if meta.get("stay_year") and meta["stay_year"] != "Not specified":
            meta_bits.append(f"Stayed in: **{meta['stay_year']}**")
        if meta_bits:
            st.markdown(" · ".join(meta_bits))

        # Stars
        st.markdown(
            f"""
            <div class="ta-stars">{star_string}</div>
            <div class="ta-stars-text">Predicted rating: <b>{stars}/5</b> based on sentiment</div>
            """,
            unsafe_allow_html=True,
        )

        # Topics
        st.markdown(
            '<div class="ta-section-title">🧩 Topics detected</div>',
            unsafe_allow_html=True,
        )
        if topics:
            chips_html = "".join(
                [f'<span class="ta-chip">{t}</span>' for t in topics]
            )
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.markdown(
                "_No explicit topics detected. The review might be too generic._"
            )

        # Cleaned review (no separate title, just the box)
        st.markdown(
            f'<div class="ta-review-box">{cleaned_review}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ta-meta">This version is generated automatically to standardize and refine the writing style.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)  # end big output card
