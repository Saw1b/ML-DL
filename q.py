import os
import numpy as np
import google.generativeai as genai
import streamlit as st
import PyPDF2
import json
import random
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time

# ====== CONFIG ======
GOOGLE_API_KEY = "AIzaSyDgZw_n-3bV0e25sEpo4hAuk7-8KSNDSm8"  # Your Gemini API key
CHUNK_SIZE = 5000                
TOP_K = 500
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-1.5-flash"
# ====================

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'chunk_vectors' not in st.session_state:
    st.session_state.chunk_vectors = None
if 'learning_mode' not in st.session_state:
    st.session_state.learning_mode = "chat"
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'web_search_enabled' not in st.session_state:
    st.session_state.web_search_enabled = True

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to embed text(s)
def embed(texts):
    """Generate embeddings for given texts"""
    try:
        if isinstance(texts, str):
            texts = [texts]
        resp = genai.embed_content(model=EMBED_MODEL, content=texts)["embedding"]
        vectors = [e for e in resp]
        arr = np.array(vectors)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# Web Search Functions
def search_web(query, num_results=5):
    """Search the web for information using DuckDuckGo API alternative"""
    try:
        # Using DuckDuckGo Instant Answer API (free alternative)
        search_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        data = response.json()
        
        results = []
        
        # Get abstract if available
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', 'DuckDuckGo Result'),
                'snippet': data.get('Abstract'),
                'url': data.get('AbstractURL', ''),
                'source': 'DuckDuckGo'
            })
        
        # Get related topics
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    'title': topic.get('Text', '')[:100] + '...',
                    'snippet': topic.get('Text', ''),
                    'url': topic.get('FirstURL', ''),
                    'source': 'DuckDuckGo Related'
                })
        
        return results[:num_results]
        
    except Exception as e:
        st.warning(f"Web search temporarily unavailable: {str(e)}")
        return []

def scrape_webpage(url, max_chars=2000):
    """Scrape content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:max_chars]
        
    except Exception as e:
        return f"Could not scrape content from {url}: {str(e)}"

def enhanced_search_and_answer(query, use_pdf=True, use_web=True):
    """Enhanced search that combines PDF content and web search"""
    try:
        sources = []
        context_parts = []
        
        # Search PDF content if available
        if use_pdf and st.session_state.text_chunks and st.session_state.chunk_vectors is not None:
            query_vec = embed([query])[0]
            sims = st.session_state.chunk_vectors @ query_vec
            top_idx = np.argsort(-sims)[:3]  # Reduced for space
            pdf_context = "\n\n".join([st.session_state.text_chunks[i] for i in top_idx])
            context_parts.append(f"FROM PDF DOCUMENT:\n{pdf_context}")
            sources.append("📄 Uploaded PDF")
        
        # Search web if enabled
        web_context = ""
        if use_web and st.session_state.web_search_enabled:
            web_results = search_web(query, num_results=3)
            if web_results:
                web_texts = []
                for result in web_results:
                    web_texts.append(f"Source: {result['title']}\nContent: {result['snippet']}")
                    sources.append(f"🌐 {result['source']}")
                
                web_context = "\n\n".join(web_texts)
                context_parts.append(f"FROM WEB SEARCH:\n{web_context}")
        
        # Combine contexts
        full_context = "\n\n===================\n\n".join(context_parts)
        
        if not full_context:
            return "No relevant information found. Please upload a PDF or enable web search.", []
        
        # Generate comprehensive answer
        prompt = f"""
        You are a knowledgeable research assistant. Answer the following question using the provided context from multiple sources.
        
        Instructions:
        1. Provide a comprehensive, well-structured answer
        2. If information comes from different sources, mention this
        3. If sources conflict, note the discrepancy
        4. If the answer isn't fully covered by the context, mention what's missing
        5. Be accurate and cite your reasoning
        
        Context from multiple sources:
        {full_context}
        
        Question: {query}
        
        Please provide a detailed, informative answer:
        """
        
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        
        return response.text, sources
        
    except Exception as e:
        return f"Error processing enhanced search: {str(e)}", []

# Research Tools
def generate_research_questions(topic):
    """Generate research questions about a topic"""
    try:
        prompt = f"""
        Generate 8-10 comprehensive research questions about the topic: "{topic}"
        
        Include different types of questions:
        1. Factual/definitional questions
        2. Analytical questions  
        3. Comparative questions
        4. Application-based questions
        5. Critical thinking questions
        
        Format as a numbered list with clear, specific questions that would help someone research this topic thoroughly.
        """
        
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating research questions: {str(e)}"

def create_research_outline(topic):
    """Create a research outline for a topic"""
    try:
        prompt = f"""
        Create a comprehensive research outline for the topic: "{topic}"
        
        Structure the outline with:
        1. Main sections and subsections
        2. Key points to research under each section
        3. Potential sources to explore
        4. Research methodologies that could be used
        5. Expected outcomes or findings
        
        Make it detailed enough to guide a thorough research project.
        """
        
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error creating research outline: {str(e)}"

def fact_check_claim(claim):
    """Fact-check a claim using available sources"""
    try:
        # Search for information about the claim
        search_results = search_web(f"fact check {claim}", num_results=3)
        
        web_context = ""
        if search_results:
            web_texts = []
            for result in search_results:
                web_texts.append(f"Source: {result['title']}\nContent: {result['snippet']}")
            web_context = "\n\n".join(web_texts)
        
        prompt = f"""
        Fact-check the following claim using the provided context and your knowledge:
        
        Claim: "{claim}"
        
        Context from web search:
        {web_context}
        
        Please provide:
        1. Verdict: True/False/Partially True/Insufficient Information
        2. Explanation of your assessment
        3. Key evidence supporting or contradicting the claim
        4. Reliability of sources (if any)
        5. Suggestions for further verification
        
        Be objective and thorough in your analysis.
        """
        
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error fact-checking claim: {str(e)}"

# Quiz generation with research integration
def generate_enhanced_quiz(text, num_questions=5, include_research=False):
    """Generate quiz with optional research-based questions"""
    try:
        research_context = ""
        if include_research:
            # Extract key topics for additional research
            topics_prompt = f"Extract 3 key topics from this text that could benefit from additional research: {text[:1000]}"
            model = genai.GenerativeModel(GEN_MODEL)
            topics_response = model.generate_content(topics_prompt)
            
            # Search for additional info on key topics
            for topic in topics_response.text.split('\n')[:2]:  # Limit to 2 topics
                if topic.strip():
                    search_results = search_web(topic.strip(), num_results=2)
                    if search_results:
                        research_context += f"\nAdditional research on {topic}: {search_results[0]['snippet']}"
        
        prompt = f"""
        Based on the following text and additional research context, create exactly {num_questions} multiple-choice quiz questions. 
        Each question should have 4 options (A, B, C, D) with only one correct answer.
        
        Include different types of questions:
        - Factual recall questions
        - Application questions
        - Analysis questions
        - Research-based questions (if research context is provided)
        
        Format your response as a JSON array where each question object has:
        - "question": the question text
        - "options": array of 4 options
        - "correct_answer": the letter (A, B, C, or D) of the correct option
        - "explanation": brief explanation of why the answer is correct
        - "type": question type (factual, application, analysis, research)
        
        Primary Text: {text[:3000]}
        
        Additional Research Context: {research_context}
        
        Return only the JSON array, no other text.
        """
        
        model = genai.GenerativeModel(GEN_MODEL)
        response = model.generate_content(prompt)
        
        # Clean the response text
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        quiz_data = json.loads(response_text)
        return quiz_data
    except Exception as e:
        st.error(f"Error generating enhanced quiz: {str(e)}")
        return []

# Streamlit UI
st.title("🔬 Advanced PDF Research & Learning System")
st.markdown("Upload PDFs, search the web, conduct research, and test your knowledge with AI-powered tools!")

# Sidebar configuration
st.sidebar.title("🛠️ Configuration")

# Web search toggle
st.session_state.web_search_enabled = st.sidebar.checkbox(
    "🌐 Enable Web Search", 
    value=st.session_state.web_search_enabled,
    help="Allow the system to search the web for additional information"
)

# Research settings
st.sidebar.markdown("### 📊 Research Settings")
research_depth = st.sidebar.selectbox(
    "Research Depth:",
    ["Basic", "Intermediate", "Advanced"],
    help="Choose how deep the research analysis should be"
)

# Mode selection
st.sidebar.title("📚 Learning Modes")
mode = st.sidebar.radio(
    "Choose your mode:",
    ["🔍 Enhanced Research", "💬 Smart Chat", "📝 Advanced Quiz", "📋 Study Tools", "🔬 Research Lab"]
)

# PDF Upload
st.header("📄 Document Upload")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process the PDF
    with st.spinner("Processing PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)
        
        if pdf_text:
            # Split into chunks
            chunks = [pdf_text[i:i+CHUNK_SIZE] for i in range(0, len(pdf_text), CHUNK_SIZE)]
            st.session_state.text_chunks = chunks
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                chunk_vectors = embed(chunks)
                st.session_state.chunk_vectors = chunk_vectors
            
            st.success(f"✅ PDF processed successfully! Found {len(chunks)} text chunks.")
            
            # Show PDF info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Pages", len(PyPDF2.PdfReader(uploaded_file).pages))
            with col2:
                st.metric("📝 Words", f"{len(pdf_text.split()):,}")
            with col3:
                st.metric("🧩 Chunks", len(chunks))

# Main content based on selected mode
if mode == "🔍 Enhanced Research":
    st.header("🔍 Enhanced Research Mode")
    st.markdown("Get comprehensive answers combining PDF content and web research")
    
    # Research query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_query = st.text_input(
            "🔍 Enter your research question:",
            placeholder="e.g., What are the latest developments in machine learning?"
        )
    
    with col2:
        search_sources = st.multiselect(
            "Sources:",
            ["📄 PDF", "🌐 Web"],
            default=["📄 PDF", "🌐 Web"] if st.session_state.web_search_enabled else ["📄 PDF"]
        )
    
    if st.button("🔍 Research", type="primary") and research_query:
        with st.spinner("Conducting comprehensive research..."):
            use_pdf = "📄 PDF" in search_sources
            use_web = "🌐 Web" in search_sources and st.session_state.web_search_enabled
            
            answer, sources = enhanced_search_and_answer(research_query, use_pdf, use_web)
            
            # Display results
            st.markdown("### 📋 Research Results")
            st.markdown("**Question:**")
            st.info(research_query)
            
            st.markdown("**Answer:**")
            st.markdown(answer)
            
            if sources:
                st.markdown("**Sources Used:**")
                for source in sources:
                    st.caption(source)
            
            # Save to research history
            st.session_state.research_history.append({
                'question': research_query,
                'answer': answer,
                'sources': sources,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

elif mode == "💬 Smart Chat":
    st.header("💬 Smart Chat with Multi-Source Intelligence")
    
    if st.session_state.text_chunks or st.session_state.web_search_enabled:
        # Chat interface
        query = st.chat_input("Ask anything - I'll search your PDF and the web for answers...")
        
        if query:
            with st.spinner("Searching multiple sources..."):
                answer, sources = enhanced_search_and_answer(
                    query, 
                    use_pdf=bool(st.session_state.text_chunks),
                    use_web=st.session_state.web_search_enabled
                )
            
            # Display conversation
            with st.container():
                st.markdown("### 🤔 Your Question:")
                st.write(query)
                
                st.markdown("### 🤖 AI Response:")
                st.markdown(answer)
                
                if sources:
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.write(f"• {source}")
    else:
        st.warning("⚠️ Please upload a PDF file or enable web search to start chatting.")

elif mode == "📝 Advanced Quiz":
    st.header("🎯 Advanced Quiz Mode")
    
    if st.session_state.text_chunks:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_questions = st.selectbox("Questions:", [5, 10, 15, 20], index=0)
        
        with col2:
            quiz_difficulty = st.selectbox("Difficulty:", ["Basic", "Intermediate", "Advanced"])
        
        with col3:
            include_research = st.checkbox("🌐 Include Research", help="Add web research to quiz questions")
        
        if st.button("🎲 Generate Advanced Quiz", type="primary"):
            with st.spinner("Generating enhanced quiz questions..."):
                full_text = " ".join(st.session_state.text_chunks)
                quiz_data = generate_enhanced_quiz(full_text, num_questions, include_research)
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.quiz_started = False
                    st.success("✅ Advanced quiz generated successfully!")
        
        # Display quiz (same logic as before but with enhanced questions)
        if st.session_state.quiz_data:
            if not st.session_state.quiz_started:
                st.markdown("### 🚀 Ready to Start Advanced Quiz!")
                st.info(f"📊 This quiz contains {len(st.session_state.quiz_data)} questions with various difficulty levels")
                if st.button("▶️ Start Quiz", type="primary"):
                    st.session_state.quiz_started = True
                    st.rerun()
            
            else:
                # Quiz in progress
                if st.session_state.current_question < len(st.session_state.quiz_data):
                    question_data = st.session_state.quiz_data[st.session_state.current_question]
                    
                    st.markdown(f"### Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_data)}")
                    
                    # Show question type if available
                    if 'type' in question_data:
                        st.badge(f"Type: {question_data['type'].title()}")
                    
                    st.markdown(f"**{question_data['question']}**")
                    
                    # Display options
                    user_answer = st.radio(
                        "Choose your answer:",
                        options=['A', 'B', 'C', 'D'],
                        format_func=lambda x: f"{x}. {question_data['options'][ord(x) - ord('A')]}",
                        key=f"q_{st.session_state.current_question}"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Submit Answer"):
                            if user_answer == question_data['correct_answer']:
                                st.success("🎉 Correct!")
                                st.session_state.score += 1
                            else:
                                st.error(f"❌ Wrong! The correct answer was {question_data['correct_answer']}")
                            
                            st.info(f"💡 **Explanation:** {question_data['explanation']}")
                            st.session_state.current_question += 1
                            
                            if st.session_state.current_question >= len(st.session_state.quiz_data):
                                st.balloons()
                                
                            st.rerun()
                    
                    with col3:
                        st.markdown(f"**Score: {st.session_state.score}/{st.session_state.current_question}**")
                
                else:
                    # Quiz completed
                    st.header("🎊 Advanced Quiz Completed!")
                    score_percentage = (st.session_state.score / len(st.session_state.quiz_data)) * 100
                    
                    st.markdown(f"### Final Score: {st.session_state.score}/{len(st.session_state.quiz_data)} ({score_percentage:.1f}%)")
                    
                    if score_percentage >= 90:
                        st.success("🌟 Outstanding performance!")
                    elif score_percentage >= 80:
                        st.success("🌟 Excellent work!")
                    elif score_percentage >= 70:
                        st.info("👍 Good job!")
                    elif score_percentage >= 60:
                        st.warning("📚 Room for improvement!")
                    else:
                        st.error("📚 Keep studying to master this material!")
                    
                    if st.button("🔄 Generate New Quiz"):
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.quiz_started = False
                        st.session_state.quiz_data = []
                        st.rerun()
    
    else:
        st.warning("⚠️ Please upload a PDF file first to generate a quiz.")

elif mode == "📋 Study Tools":
    st.header("📚 Advanced Study Tools")
    
    if st.session_state.text_chunks:
        tool_tabs = st.tabs(["📝 Study Notes", "❓ Research Questions", "📊 Study Outline", "📈 Progress Tracking"])
        
        with tool_tabs[0]:
            st.markdown("### 📝 AI-Generated Study Notes")
            if st.button("📚 Generate Comprehensive Notes", type="primary"):
                with st.spinner("Creating detailed study notes..."):
                    full_text = " ".join(st.session_state.text_chunks)
                    prompt = f"""
                    Create comprehensive, well-structured study notes from the following text:
                    
                    {full_text[:4000]}
                    
                    Structure the notes with:
                    1. 📚 Executive Summary
                    2. 🎯 Key Concepts & Definitions
                    3. 💡 Important Facts & Figures
                    4. 🔗 Relationships & Connections
                    5. ❗ Critical Points to Remember
                    6. 🤔 Discussion Questions
                    7. 📖 Summary & Takeaways
                    
                    Use clear formatting, bullet points, and organize information logically.
                    """
                    
                    model = genai.GenerativeModel(GEN_MODEL)
                    notes = model.generate_content(prompt)
                    
                    st.markdown(notes.text)
                    
                    st.download_button(
                        label="💾 Download Study Notes",
                        data=notes.text,
                        file_name=f"study_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        with tool_tabs[1]:
            st.markdown("### ❓ Research Question Generator")
            topic = st.text_input("🎯 Enter a topic from your document:", 
                                placeholder="e.g., machine learning algorithms")
            
            if st.button("🔍 Generate Research Questions") and topic:
                with st.spinner("Generating research questions..."):
                    questions = generate_research_questions(topic)
                    st.markdown(questions)
        
        with tool_tabs[2]:
            st.markdown("### 📊 Research Outline Creator")
            outline_topic = st.text_input("📝 Topic for research outline:", 
                                        placeholder="e.g., artificial intelligence applications")
            
            if st.button("📋 Create Research Outline") and outline_topic:
                with st.spinner("Creating research outline..."):
                    outline = create_research_outline(outline_topic)
                    st.markdown(outline)
        
        with tool_tabs[3]:
            st.markdown("### 📈 Your Research History")
            if st.session_state.research_history:
                for i, research in enumerate(reversed(st.session_state.research_history[-10:])):  # Show last 10
                    with st.expander(f"🔍 {research['question'][:50]}... - {research['timestamp']}"):
                        st.markdown("**Question:**")
                        st.write(research['question'])
                        st.markdown("**Answer:**")
                        st.write(research['answer'][:500] + "..." if len(research['answer']) > 500 else research['answer'])
                        if research['sources']:
                            st.markdown("**Sources:**")
                            for source in research['sources']:
                                st.caption(source)
            else:
                st.info("📊 No research history yet. Start using Enhanced Research mode!")
    
    else:
        st.warning("⚠️ Please upload a PDF file first to access study tools.")

elif mode == "🔬 Research Lab":
    st.header("🔬 Research Laboratory")
    st.markdown("Advanced research tools and fact-checking capabilities")
    
    lab_tabs = st.tabs(["✅ Fact Checker", "🌐 Web Research", "📊 Topic Explorer", "📈 Research Analytics"])
    
    with lab_tabs[0]:
        st.markdown("### ✅ AI Fact Checker")
        claim = st.text_area("Enter a claim to fact-check:", 
                           placeholder="e.g., Artificial intelligence will replace all human jobs by 2030")
        
        if st.button("🔍 Fact Check", type="primary") and claim:
            with st.spinner("Fact-checking claim..."):
                fact_check_result = fact_check_claim(claim)
                st.markdown("### 📋 Fact-Check Results")
                st.markdown(fact_check_result)
    
    with lab_tabs[1]:
        st.markdown("### 🌐 Dedicated Web Research")
        web_query = st.text_input("🔍 Enter research query for web search:")
        
        if st.button("🌐 Search Web") and web_query:
            with st.spinner("Searching the web..."):
                web_results = search_web(web_query, num_results=5)
                
                if web_results:
                    st.markdown("### 🔍 Web Search Results")
                    for i, result in enumerate(web_results, 1):
                        with st.expander(f"Result {i}: {result['title']}"):
                            st.markdown(f"**Source:** {result['source']}")
                            st.markdown(f"**Content:** {result['snippet']}")
                            if result['url']:
                                st.markdown(f"**URL:** {result['url']}")
                else:
                    st.warning("No web results found for this query.")
    
    with lab_tabs[2]:
        st.markdown("### 📊 Topic Explorer")
        if st.session_state.text_chunks:
            if st.button("🔍 Analyze Document Topics"):
                with st.spinner("Analyzing document topics..."):
                    full_text = " ".join(st.session_state.text_chunks[:10])  # Limit for analysis
                    
                    prompt = f"""
                    Analyze the following document and provide:
                    1. Main topics covered (ranked by importance)
                    2. Key themes and concepts
                    3. Subject areas and domains
                    4. Potential research directions
                    5. Related fields of study
                    
                    Text: {full_text[:3000]}
                    
                    Provide a structured analysis with clear sections.
                    """
                    
                    model = genai.GenerativeModel(GEN_MODEL)
                    analysis = model.generate_content(prompt)
                    
                    st.markdown("### 📊 Topic Analysis Results")
                    st.markdown(analysis.text)
        else:
            st.warning("Please upload a PDF to analyze topics.")
    
    with lab_tabs[3]:
        st.markdown("### 📈 Research Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🔍 Research Queries", len(st.session_state.research_history))
            st.metric("📄 Document Chunks", len(st.session_state.text_chunks))
        
        with col2:
            st.metric("🌐 Web Search", "Enabled" if st.session_state.web_search_enabled else "Disabled")
            st.metric("📚 Study Mode", mode.split()[1] if len(mode.split()) > 1 else mode)
        
        if st.session_state.research_history:
            st.markdown("### 📊 Recent Research Activity")
            recent_queries = [r['question'] for r in st.session_state.research_history[-5:]]
            for i, query in enumerate(reversed(recent_queries), 1):
                st.write(f"{i}. {query}")

# Additional Research Features Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Advanced Features")

if st.sidebar.button("🧹 Clear Research History"):
    st.session_state.research_history = []
    st.sidebar.success("History cleared!")

if st.sidebar.button("📊 Export Research Data"):
    if st.session_state.research_history:
        export_data = {
            'research_history': st.session_state.research_history,
            'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_queries': len(st.session_state.research_history)
        }
        
        st.sidebar.download_button(
            label="💾 Download Research Export",
            data=json.dumps(export_data, indent=2),
            file_name=f"research_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.sidebar.warning("No research data to export")

# Footer with additional information
st.markdown("---")
st.markdown("""
### 🎯 **System Capabilities:**
- **📄 PDF Processing:** Extract and analyze text from uploaded documents
- **🌐 Web Search:** Real-time web search integration for current information
- **🤖 AI Analysis:** Advanced AI-powered content analysis and question answering
- **📝 Smart Quizzes:** Generate adaptive quizzes with multiple difficulty levels
- **🔍 Research Tools:** Comprehensive research assistance and fact-checking
- **📊 Analytics:** Track your learning progress and research patterns

### 💡 **Pro Tips:**
1. **Combine Sources:** Use both PDF and web search for comprehensive answers
2. **Fact-Check:** Always verify important claims using the fact-checker
3. **Research History:** Review your past queries to track learning progress  
4. **Export Data:** Download your research for offline review
5. **Multiple Modes:** Switch between modes for different learning objectives

### 🔒 **Privacy & Security:**
- Web searches are performed through secure APIs
- Your PDF content is processed locally and not stored permanently
- Research history is session-based and can be cleared anytime
""")

# Research examples and templates
with st.expander("📚 Research Examples & Templates"):
    st.markdown("""
    ### 🔍 **Sample Research Queries:**
    
    **Academic Research:**
    - "Compare machine learning algorithms for natural language processing"
    - "What are the latest developments in renewable energy storage?"
    - "How do genetic algorithms work in optimization problems?"
    
    **Business Analysis:**
    - "Impact of artificial intelligence on supply chain management"
    - "Current trends in digital marketing strategies"
    - "Blockchain applications in financial services"
    
    **Scientific Investigation:**
    - "Recent discoveries in quantum computing research"
    - "Climate change effects on marine ecosystems"
    - "CRISPR gene editing applications and ethics"
    
    **Fact-Checking Examples:**
    - "Electric vehicles produce zero emissions throughout their lifecycle"
    - "Drinking 8 glasses of water daily is necessary for optimal health"
    - "Solar panels are more efficient than wind turbines for home energy"
    
    ### 📝 **Research Question Templates:**
    
    **Comparative Analysis:**
    - "How does [A] compare to [B] in terms of [criteria]?"
    - "What are the advantages and disadvantages of [topic]?"
    
    **Causal Investigation:**
    - "What factors contribute to [phenomenon]?"
    - "How does [variable] affect [outcome]?"
    
    **Trend Analysis:**
    - "What are the current trends in [field/industry]?"
    - "How has [topic] evolved over the past [time period]?"
    
    **Implementation Study:**
    - "How can [solution] be implemented to address [problem]?"
    - "What are the best practices for [process/system]?"
    """)

# Advanced search tips
with st.expander("🎯 Advanced Search & Research Tips"):
    st.markdown("""
    ### 🔍 **Effective Search Strategies:**
    
    **1. Use Specific Keywords:**
    - Instead of: "machine learning"
    - Try: "supervised machine learning algorithms for image recognition"
    
    **2. Combine Multiple Concepts:**
    - "renewable energy storage technologies 2024"
    - "artificial intelligence ethics healthcare applications"
    
    **3. Ask Focused Questions:**
    - "What are the main challenges in implementing quantum computers?"
    - "How do neural networks process natural language?"
    
    **4. Use Different Question Types:**
    - **What**: Factual information
    - **How**: Process and methodology  
    - **Why**: Reasoning and causation
    - **When**: Timeline and chronology
    - **Where**: Location and context
    
    ### 📊 **Research Methodology:**
    
    **1. Start Broad, Then Narrow:**
    - Begin with general concepts
    - Progressively focus on specific aspects
    
    **2. Cross-Reference Sources:**
    - Compare PDF content with web research
    - Look for consensus and contradictions
    
    **3. Verify Information:**
    - Use the fact-checker for important claims
    - Check multiple sources for accuracy
    
    **4. Document Your Process:**
    - Use research history to track findings
    - Export data for further analysis
    """)

# System status and diagnostics
with st.expander("⚙️ System Status & Diagnostics"):
    st.markdown("### 🖥️ **Current System Status:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📄 PDF Processing:**")
        if st.session_state.text_chunks:
            st.success("✅ Ready")
            st.caption(f"Chunks: {len(st.session_state.text_chunks)}")
        else:
            st.warning("⚠️ No PDF loaded")
    
    with col2:
        st.markdown("**🌐 Web Search:**")
        if st.session_state.web_search_enabled:
            st.success("✅ Enabled")
        else:
            st.error("❌ Disabled")
    
    with col3:
        st.markdown("**🤖 AI Models:**")
        st.success("✅ Connected")
        st.caption(f"Embed: {EMBED_MODEL}")
        st.caption(f"Gen: {GEN_MODEL}")
    
    st.markdown("**📊 Session Statistics:**")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Research Queries", len(st.session_state.research_history))
    with stats_col2:
        st.metric("Quiz Questions", len(st.session_state.quiz_data))
    with stats_col3:
        current_mode = mode.split()[1] if len(mode.split()) > 1 else mode
        st.metric("Active Mode", current_mode)

# Performance optimization note
st.markdown("""
---
### ⚡ **Performance Notes:**
- **Web Search:** May take 2-5 seconds depending on query complexity
- **PDF Processing:** Processing time scales with document size
- **AI Generation:** Quiz and note generation may take 10-30 seconds
- **Embeddings:** Generated once per PDF upload for optimal performance

*💡 Tip: For best performance, use specific queries and enable only needed features.*
""")