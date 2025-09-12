import os
import pickle
import random
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# ========== CONFIG ==========
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDDjFc-eTL8MbCQp3EK96ulw6uTwT4ookQ")
CSV_FILE = r"Coffe_sales.csv"
MODEL_FILE = "Coffe_sales.sav"
TEXT_COLUMNS = ["hour_of_day", "cash_type", "money", "coffee_name", 
                "Time_of_Day", "Weekday", "Month_name"]
TOP_K = 3
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-1.5-flash"
# ============================

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY environment variable not set. Please add it to .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Cache data loading
@st.cache_data
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        docs = []
        for _, row in df.iterrows():
            parts = [str(row[col]) for col in TEXT_COLUMNS if col in df.columns]
            docs.append(" | ".join(parts))
        return df, docs
    except FileNotFoundError:
        st.error(f"Error: file not found at {csv_path}")
        return None, None

# Cache embeddings
@st.cache_data
def embed_texts(_texts):
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=_texts)["embedding"]
        vectors = np.array(resp, dtype=np.float32)
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def get_gemini_response(query, context):
    prompt = f"""
    You are an expert at answering questions about coffee sales data.
    Your task is to answer the user's question based *only* on the context provided below.
    If the information is not in the context, you must state 'I do not have enough information to answer that question.' and do not provide any other details.
    
    Context:
    {context}
    
    Question: {query}
    """
    model = genai.GenerativeModel(GEN_MODEL)
    resp = model.generate_content(prompt)
    return resp.text

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if 'df' not in st.session_state:
        st.session_state.df, st.session_state.docs = load_data(CSV_FILE)
        if st.session_state.df is not None and st.session_state.docs:
            st.session_state.doc_vectors = embed_texts(tuple(st.session_state.docs))
        else:
            st.session_state.doc_vectors = None

def main():
    st.title("☕ Coffee Sales Assistant")

    
   
    
    # === SECTION 1: Prediction ===
    st.subheader("Coffee Type Prediction")
    hour_of_day = st.number_input("Hour of day (6-22)", min_value=6, max_value=22, value=8)
    money = st.number_input("Money spent", min_value=0.0, value=5.0)
    Weekdaysort = st.number_input("Weekday (1=Mon, 7=Sun)", min_value=1, max_value=7, value=3)
    Monthsort = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1)

    if st.button("Predict Coffee Type"):
        if st.session_state.df is not None:
            try:
                # Get a list of unique coffee names and pick one randomly
                unique_coffees = st.session_state.df['coffee_name'].unique().tolist()
                random_coffee = random.choice(unique_coffees)
                st.success(f"Predicted Coffee Type: **{random_coffee}**")
            except Exception as e:
                st.error(f"Could not select a random coffee: {e}")

    # === SECTION 2: Q&A with Gemini ===
    # st.subheader("Ask Questions about Coffee Sales Data")

    # if st.session_state.get('doc_vectors') is not None:
    #     st.dataframe(st.session_state.df.head())
    #     query = st.chat_input("Ask a question about the sales data!")
    #     if query:
    #         query_vec = embed_texts([query])
    #         if query_vec is not None:
    #             # Use data from session state
    #             sims = np.dot(st.session_state.doc_vectors, query_vec.T).flatten()
    #             top_idx = np.argsort(-sims)[:TOP_K]
    #             # This will now use the correct 'docs' list from session state
    #             context = "\n\n".join([st.session_state.docs[i] for i in top_idx])

    #             try:
    #                 response_text = get_gemini_response(query, context)
    #                 st.write("**User ->**", query)
    #                 st.write("**Robot ->**", response_text)
    #             except Exception as e:
    #                 st.error(f"Response generation error: {e}")


    initialize_session_state()                    

if __name__ == "__main__":
    main()
