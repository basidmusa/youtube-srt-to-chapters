import streamlit as st
import re
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# --- App Configuration ---
st.set_page_config(page_title="AI YouTube Chapters Pro", page_icon="📝")

@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_ai_title(api_key, text_segment):
    """Generates a smart title using Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Summarize this transcript segment into a 3-5 word catchy YouTube chapter title. Return ONLY the title:\n\n{text_segment[:2000]}"
        response = model.generate_content(prompt)
        return response.text.strip().replace('"', '')
    except:
        return " ".join(text_segment.split()[:5]) + "..."

def srt_time_to_yt(srt_time):
    parts = srt_time.split(',')[0].split(':')
    h, m, s = parts[0], parts[1], parts[2]
    return f"{int(h)}:{m}:{s}" if int(h) > 0 else f"{m}:{s}"

def parse_srt(content):
    pattern = re.compile(r'\d+\s+(\d{2}:\d{2}:\d{2},\d{3}) --> .+\s+((?:(?!\d+\s+\d{2}:).*\n?)+)')
    matches = pattern.findall(content)
    times = [m[0] for m in matches]
    texts = [m[1].replace('\n', ' ').strip() for m in matches]
    return times, texts

# --- Sidebar ---
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.markdown("[Get a free key here](https://aistudio.google.com/)")

# --- Main UI ---
st.title("📝 AI YouTube Chapter Generator")
st.info("Upload your SRT, and I'll find the topics and name them for you.")

uploaded_file = st.file_uploader("Upload SRT File", type="srt")

if uploaded_file:
    content = uploaded_file.getvalue().decode("utf-8")
    times, texts = parse_srt(content)
    
    col1, col2 = st.columns(2)
    with col1:
        sensitivity = st.slider("Topic Detection Sensitivity", 0.1, 0.8, 0.45, help="Lower = more frequent changes.")
    with col2:
        min_gap = st.number_input("Minimum lines per chapter", value=20)

    if st.button("🚀 Generate Smart Chapters"):
        if not api_key:
            st.error("Please add your Gemini API Key in the sidebar!")
        else:
            with st.spinner("Analyzing themes and writing titles..."):
                nlp_model = load_nlp_model()
                embeddings = nlp_model.encode(texts, convert_to_tensor=True)
                
                chapters = []
                last_idx = 0
                
                for i in range(min_gap, len(embeddings) - min_gap):
                    prev = embeddings[i-min_gap:i].mean(dim=0)
                    next_v = embeddings[i:i+min_gap].mean(dim=0)
                    similarity = util.cos_sim(prev, next_v).item()
                    
                    if (similarity < sensitivity and (i - last_idx) > min_gap) or i == min_gap:
                        ts = "00:00" if i == min_gap else srt_time_to_yt(times[i])
                        context_chunk = " ".join(texts[i : i+35])
                        title = get_ai_title(api_key, context_chunk)
                        chapters.append(f"{ts} {title}")
                        last_idx = i

                # Final Output
                result_text = "\n".join(chapters)
                st.subheader("Results")
                st.text_area("YouTube Ready Format:", value=result_text, height=300)
                
                # --- DOWNLOAD BUTTON ---
                st.download_button(
                    label="💾 Download as Text File",
                    data=result_text,
                    file_name="youtube_chapters.txt",
                    mime="text/plain"
                )
                st.success("All set! Just copy-paste these into your video description.")
