import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st

# ===== CONFIG =====
GOOGLE_API_KEY = "AIzaSyDZFIqTOZ_n8RXC2nRM_bMY6ZUufD-XUnM"   # Gemini API key
CSV_FILE = "bi.csv"                    # Path of file
TEXT_COLUMNS = ["age", "gender","Name","country","residence","entryEXAM", "prevEducation", "studyHOURS"]    # Columns to use for retrieval
TOP_K = 3
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-1.5-flash"
# ==================

genai.configure(api_key=GOOGLE_API_KEY)

# 1. Load CSV with encoding handling
@st.cache_data
def load_csv_with_encoding(file_path):
    """Load CSV file trying different encodings"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            st.success(f"Successfully loaded CSV with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error with {encoding}: {str(e)}")
            continue
    
    # If all encodings fail, try with error handling
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        st.warning("Loaded CSV with UTF-8 encoding, ignoring decode errors")
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {str(e)}")
        return None

# Load the dataframe
df = load_csv_with_encoding(CSV_FILE)

if df is not None:
    st.write("### Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    
    # 2. Build documents from chosen columns
    docs = []
    available_columns = [col for col in TEXT_COLUMNS if col in df.columns]
    
    if not available_columns:
        st.error(f"None of the specified columns {TEXT_COLUMNS} found in the CSV")
        st.write("Available columns:", list(df.columns))
    else:
        st.write(f"Using columns: {available_columns}")
        
        for _, row in df.iterrows():
            parts = []
            for col in available_columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                if value:  # Only add non-empty values
                    parts.append(f"{col}: {value}")
            docs.append(" | ".join(parts))

        # 3. Function to embed texts
        @st.cache_data
        def embed(texts):
            """Embed texts using Gemini API"""
            try:
                resp = genai.embed_content(model=EMBED_MODEL, content=texts)
                vectors = resp["embedding"]
                arr = np.array(vectors, dtype=np.float32)
                # Normalize vectors
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                return arr / (norms + 1e-10)
            except Exception as e:
                st.error(f"Embedding error: {str(e)}")
                return None

        # 4. Embed all documents
        with st.spinner("Embedding documents..."):
            doc_vectors = embed(docs)

        if doc_vectors is not None:
            st.success(f"Successfully embedded {len(docs)} documents")
            
            # User query interface
            def process_query(query):
                """Process user query and return response"""
                if not query or query.strip() == "":
                    return
                
                try:
                    # Embed query
                    query_vec = embed([query])
                    if query_vec is None:
                        st.error("Failed to embed query")
                        return
                    
                    query_vec = query_vec[0]
                    
                    # Calculate similarities
                    similarities = doc_vectors @ query_vec
                    top_indices = np.argsort(-similarities)[:TOP_K]
                    
                    # Get top contexts
                    top_contexts = []
                    for i, idx in enumerate(top_indices):
                        score = similarities[idx]
                        context = docs[idx]
                        top_contexts.append(f"Document {i+1} (Score: {score:.3f}):\n{context}")
                    
                    context = "\n\n".join(top_contexts)
                    
                    # Generate response
                    prompt = f"""Answer the question using only the provided context. Be specific and cite relevant information from the context.

Context:
{context}

Question: {query}

Answer:"""
                    
                    model = genai.GenerativeModel(GEN_MODEL)
                    response = model.generate_content(prompt)
                    
                    # Display results
                    st.write("### Query Results")
                    st.write(f"**User:** {query}")
                    st.write(f"**Assistant:** {response.text}")
                    
                    # Show retrieved context
                    with st.expander("Show Retrieved Context"):
                        st.write(context)
                        
                except Exception as e:
                    st.error(f"Query processing error: {str(e)}")

            # Chat interface
            st.write("### Ask Questions About Your Data")
            
            # Example queries
            if st.button("Show Example Queries"):
                st.write("Example queries you can try:")
                st.write("- What is the average age of students?")
                st.write("- How many students are from each country?")
                st.write("- What are the different education backgrounds?")
                st.write("- Show me students with high study hours")
            
            # Main chat input
            user_query = st.chat_input("Enter your query about the data...")
            
            if user_query:
                process_query(user_query)
                
        else:
            st.error("Failed to embed documents")
else:
    st.error("Could not load the CSV file. Please check the file path and encoding.")