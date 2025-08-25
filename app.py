import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from file
def load_dataset():
    if os.path.exists("persona_data.txt"):
        with open("persona_data.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            qa_pairs = [line.strip().split("||") for line in lines if "||" in line]
            questions, answers = zip(*qa_pairs)
            return list(questions), list(answers)
    return [], []

# Load the model (vectorizer and cosine similarity base)
def load_model(questions):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

# Get the most relevant answer using cosine similarity
def get_answer(user_input, vectorizer, tfidf_matrix, answers, questions):
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix)
    best_match_idx = sim_scores.argmax()
    confidence = sim_scores[0][best_match_idx]
    if confidence > 0.2:
        return answers[best_match_idx]
    else:
        return "😔 Sorry, I don't have any data to answer that yet. Try asking something else."

# App UI
def main():
    st.set_page_config(page_title="PersonaSynth", page_icon="🤖", layout="centered")
    st.markdown("""
        <h1 style='text-align: center;'>🤖 PersonaSynth</h1>
        <p style='text-align: center;'>Chat with your own AI avatar trained on your personality</p>
        <hr>
    """, unsafe_allow_html=True)

    questions, answers = load_dataset()

    if not questions:
        st.warning("No data found in persona_data.txt. Please add Q&A data to get started.")
        return

    vectorizer, tfidf_matrix = load_model(questions)

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712078.png", width=100)
        st.markdown("""
        ### About PersonaSynth
        Built by **Neeti Malu** to simulate intelligent, context-aware personality conversations.
        
        - 📚 1000+ curated Q&A pairs
        - ⚡ Fast, offline, no API needed
        - 🎯 Built for placements, demos, and projects
        
        --
        [💼 Connect on LinkedIn](https://www.linkedin.com/in/neeti-malu)
        """)

    # Welcome message
    st.success("👋 Welcome to PersonaSynth! Type a question below to chat with your AI twin.")

    # Search bar and Reset
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Ask me anything about myself:", key="query")
    with col2:
        if st.button("🔄 Reset"):
            st.experimental_rerun()

    if query:
        with st.spinner("Thinking like you..."):
            response = get_answer(query, vectorizer, tfidf_matrix, answers, questions)
            st.markdown(f"**🧠 Answer:** {response}")

    st.markdown("""
    <hr>
    <p style='text-align: center;'>🔧 Built by <b>Neeti Malu</b> | Powered by <i>Streamlit, Scikit-learn & TF-IDF</i></p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



