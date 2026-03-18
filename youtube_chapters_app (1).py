import sys
import subprocess
import importlib

# ---------------------------------------------------------------------------
# Environment guard — auto-installs missing packages into the SAME Python
# that is running this script, solving the pip/streamlit env mismatch.
# ---------------------------------------------------------------------------
REQUIRED = {
    "streamlit": "streamlit",
    "google.generativeai": "google-generativeai",
    "sentence_transformers": "sentence-transformers",
}

for module_name, pip_name in REQUIRED.items():
    if importlib.util.find_spec(module_name.split(".")[0]) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name],
            stdout=subprocess.DEVNULL,
        )

# ---------------------------------------------------------------------------
# Safe imports (after guard above)
# ---------------------------------------------------------------------------
import re
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI YouTube Chapters Pro", page_icon="📝")

# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_nlp_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------
def parse_srt(content: str) -> tuple[list[str], list[str]]:
    """
    Robust SRT parser.
    Splits on blank lines so every block is handled independently.
    """
    blocks = re.split(r"\n\s*\n", content.strip())
    times, texts = [], []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s+-->", lines[1])
        if match:
            times.append(match.group(1))
            texts.append(" ".join(lines[2:]).strip())
    return times, texts


def srt_time_to_yt(srt_time: str) -> str:
    """Convert '01:23:45,678' to '1:23:45' or '23:45'."""
    hms = srt_time.split(",")[0]
    h, m, s = hms.split(":")
    return f"{int(h)}:{m}:{s}" if int(h) > 0 else f"{m}:{s}"

# ---------------------------------------------------------------------------
# AI title generation
# ---------------------------------------------------------------------------
def get_ai_title(api_key: str, text_segment: str) -> str:
    """Call Gemini for a short chapter title. Falls back gracefully."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Summarize this transcript segment into a 3-5 word catchy YouTube "
            "chapter title. Return ONLY the title, no quotes:\n\n"
            f"{text_segment[:2000]}"
        )
        response = model.generate_content(prompt)
        return response.text.strip().replace('"', "")
    except Exception as exc:
        st.warning(f"AI title failed: {exc}")
        return " ".join(text_segment.split()[:5]) + "..."

# ---------------------------------------------------------------------------
# Chapter detection
# ---------------------------------------------------------------------------
def detect_chapters(
    times: list[str],
    texts: list[str],
    api_key: str,
    sensitivity: float,
    min_gap: int,
) -> list[str]:
    """
    Slide a window over sentence embeddings, detect topic shifts,
    then call Gemini to name each chapter.
    Always inserts 00:00 as the first chapter.
    """
    nlp_model = load_nlp_model()
    embeddings = nlp_model.encode(texts, convert_to_tensor=True)

    intro_chunk = " ".join(texts[:min_gap])
    intro_title = get_ai_title(api_key, intro_chunk)
    chapters = [f"00:00 {intro_title}"]
    last_idx = 0

    for i in range(min_gap, len(embeddings) - min_gap):
        prev_window = embeddings[i - min_gap : i].mean(dim=0)
        next_window = embeddings[i : i + min_gap].mean(dim=0)
        similarity = util.cos_sim(prev_window, next_window).item()

        if similarity < sensitivity and (i - last_idx) > min_gap:
            ts = srt_time_to_yt(times[i])
            context_chunk = " ".join(texts[i : i + 35])
            title = get_ai_title(api_key, context_chunk)
            chapters.append(f"{ts} {title}")
            last_idx = i

    return chapters

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.markdown("[Get a free key here](https://aistudio.google.com/)")
st.sidebar.info(
    "**Sensitivity:** lower values detect more topic changes.\n\n"
    "**Min lines/chapter:** prevents very short chapters."
)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📝 AI YouTube Chapter Generator")
st.caption("Upload your SRT subtitle file and get ready-to-paste YouTube chapters.")

uploaded_file = st.file_uploader("Upload SRT File", type="srt")

if uploaded_file:
    content = uploaded_file.getvalue().decode("utf-8")
    times, texts = parse_srt(content)

    if not times:
        st.error("Could not parse any subtitles. Check that the file is a valid SRT.")
        st.stop()

    st.success(f"Loaded {len(texts)} subtitle lines.")

    col1, col2 = st.columns(2)
    with col1:
        sensitivity = st.slider(
            "Topic Detection Sensitivity",
            min_value=0.1, max_value=0.8, value=0.45, step=0.05,
            help="Lower = more frequent chapter breaks.",
        )
    with col2:
        min_gap = st.number_input(
            "Minimum lines per chapter", min_value=5, max_value=100, value=20
        )

    if st.button("Generate Smart Chapters"):
        if not api_key:
            st.error("Please add your Gemini API Key in the sidebar first.")
            st.stop()

        if len(texts) < min_gap * 2:
            st.error(
                f"File too short ({len(texts)} lines) for the chosen gap ({min_gap}). "
                "Lower 'Minimum lines per chapter' and try again."
            )
            st.stop()

        with st.spinner("Detecting topics and writing chapter titles..."):
            chapters = detect_chapters(times, texts, api_key, sensitivity, min_gap)

        if not chapters:
            st.warning("No topic shifts detected. Try lowering the sensitivity slider.")
        else:
            result_text = "\n".join(chapters)
            st.subheader(f"Results — {len(chapters)} chapters")
            st.text_area("YouTube-ready format:", value=result_text, height=300)
            st.download_button(
                label="Download as .txt",
                data=result_text,
                file_name="youtube_chapters.txt",
                mime="text/plain",
            )
            st.success("Done! Paste these directly into your YouTube video description.")
