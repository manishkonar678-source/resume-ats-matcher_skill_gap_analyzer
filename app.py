"""
Smart Resume ATS Matcher & Skill Gap Analyzer (Text-only version)
-------------------------------------------------------------------
A simple Python + Streamlit web app that:
1. Takes resume text (pasted directly, no PDF upload needed)
2. Takes a job description (pasted text)
3. Extracts skills from both using a keyword-based skill database
4. Computes an ATS Match Score using TF-IDF + Cosine Similarity (scikit-learn)
5. Shows matched skills, missing skills (skill gap) and simple suggestions

Only dependency used for matching/scoring: scikit-learn.
Author: <Student Name>
"""

import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1. SKILL DATABASE
# A simple, fixed list of common skills used for keyword matching.
# This keeps the project "simple Python" (no heavy NLP models).
# -------------------------------------------------------------------
SKILL_DB = [
    # Programming languages
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "r",
    "go", "rust", "php", "kotlin", "swift", "html", "css",
    # Data / ML
    "machine learning", "deep learning", "data analysis", "data science",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "computer vision", "data visualization", "power bi", "tableau",
    "statistics", "excel", "matplotlib", "seaborn",
    # Web / software
    "react", "angular", "node.js", "django", "flask", "streamlit",
    "rest api", "git", "github", "docker", "kubernetes", "linux",
    "aws", "azure", "gcp", "firebase", "mongodb", "mysql", "postgresql",
    # Soft / general
    "communication", "teamwork", "leadership", "problem solving",
    "project management", "time management", "agile", "scrum",
    "critical thinking", "presentation",
]


# -------------------------------------------------------------------
# 2. TEXT CLEANING
# -------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase and normalise whitespace/punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\#\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------------------
# 3. SKILL EXTRACTION
# -------------------------------------------------------------------
def extract_skills(text: str) -> set:
    """Return the set of known skills (from SKILL_DB) found in the text."""
    cleaned = clean_text(text)
    found = set()
    for skill in SKILL_DB:
        # word-boundary-safe search so "r" doesn't match inside other words
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, cleaned):
            found.add(skill)
    return found


# -------------------------------------------------------------------
# 4. ATS MATCH SCORE (TF-IDF + Cosine Similarity, scikit-learn only)
# -------------------------------------------------------------------
def compute_ats_score(resume_text: str, jd_text: str) -> float:
    """
    Computes overall textual similarity between resume and job description
    using TF-IDF vectors and cosine similarity. Returned as a percentage.
    """
    documents = [clean_text(resume_text), clean_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


def compute_skill_score(matched: set, jd_skills: set) -> float:
    """Percentage of job-description skills that are present in the resume."""
    if not jd_skills:
        return 0.0
    return round((len(matched) / len(jd_skills)) * 100, 2)


# -------------------------------------------------------------------
# 5. STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Smart Resume ATS Matcher", page_icon="📄", layout="centered")

st.title("📄 Smart Resume ATS Matcher & Skill Gap Analyzer")
st.write(
    "Paste your resume text and a job description below to see how well "
    "your resume matches, which skills you already cover, and which ones "
    "you're missing."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Paste Resume Text")
    resume_text_input = st.text_area("Paste your resume text here", height=280)

with col2:
    st.subheader("2. Paste Job Description")
    jd_text_input = st.text_area("Paste the job description here", height=280)

analyze_clicked = st.button("🔍 Analyze Match", type="primary")

if analyze_clicked:
    if not resume_text_input.strip() or not jd_text_input.strip():
        st.error("Please paste both your resume text and a job description.")
    else:
        resume_skills = extract_skills(resume_text_input)
        jd_skills = extract_skills(jd_text_input)

        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills

        ats_text_score = compute_ats_score(resume_text_input, jd_text_input)
        skill_score = compute_skill_score(matched_skills, jd_skills)
        overall_score = round((ats_text_score + skill_score) / 2, 2)

        st.markdown("---")
        st.subheader("📊 Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall ATS Score", f"{overall_score}%")
        m2.metric("Text Similarity", f"{ats_text_score}%")
        m3.metric("Skill Coverage", f"{skill_score}%")

        st.progress(min(int(overall_score), 100))

        st.markdown("### ✅ Matched Skills")
        if matched_skills:
            st.success(", ".join(sorted(matched_skills)))
        else:
            st.info("No matching skills found.")

        st.markdown("### ⚠️ Missing Skills (Skill Gap)")
        if missing_skills:
            st.warning(", ".join(sorted(missing_skills)))
            st.write(
                "💡 Consider adding relevant experience, projects, or certifications "
                "for the skills above to improve your ATS match score."
            )
        else:
            st.success("Great! Your resume covers all the detected skills from the job description.")

        st.markdown("---")
        st.caption(
            "Note: This tool uses keyword-based skill matching and TF-IDF text "
            "similarity — it is meant as a helpful guide, not a guaranteed ATS outcome."
        )

st.markdown("---")
st.caption("Built with Python + Streamlit + scikit-learn | Smart Resume ATS Matcher & Skill Gap Analyzer")
