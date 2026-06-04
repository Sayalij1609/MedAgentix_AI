# ============================================================
# STEP 3: STREAMLIT WEB APPLICATION — app.py
# ============================================================
# PURPOSE:
#   This is the USER INTERFACE of MedAgentixAI. It provides a
#   beautiful, interactive web application where users can:
#   
#   1. Type medical questions in natural language
#   2. Filter by disease category and severity level
#   3. See AI-generated answers powered by Gemini
#   4. View the source knowledge chunks with relevance scores
#   5. Explore knowledge base statistics
#
# RUN THIS APP:
#   streamlit run app.py
#
# WHAT IS STREAMLIT?
# ==================
#   Streamlit is a Python library that turns Python scripts into
#   interactive web apps. You write Python, and Streamlit creates
#   the HTML, CSS, and JavaScript automatically. It's perfect for
#   data science and AI applications.
# ============================================================


# ============================================================
# STEP 3.1: IMPORT LIBRARIES
# ============================================================

import streamlit as st           # Step 3.1a: Streamlit for building the web interface
                                 #            st.* functions create UI elements (buttons, text, etc.)

from rag_system import MedRAG    # Step 3.1b: Import our RAG system from the rag_system.py file
                                 #            This handles all the search and AI generation

import time                      # Step 3.1c: For adding slight delays (loading animations)


# ============================================================
# STEP 3.2: PAGE CONFIGURATION
# ============================================================
# This MUST be the first Streamlit command in the script.
# It sets the browser tab title, icon, and layout.
# ============================================================

st.set_page_config(
    page_title="MedAgentixAI — Medical Knowledge RAG",     # Step 3.2a: Browser tab title
    page_icon="🏥",                                         # Step 3.2b: Browser tab icon (emoji)
    layout="wide",                                          # Step 3.2c: Use the full width of the screen
    initial_sidebar_state="expanded"                        # Step 3.2d: Sidebar starts open
)


# ============================================================
# STEP 3.3: CUSTOM CSS STYLING
# ============================================================
# Streamlit allows injecting custom CSS to make the app look
# more professional and visually stunning. We use a dark medical
# theme with gradient accents and modern typography.
# ============================================================

st.markdown("""
<style>
    /* Step 3.3a: Import Google Fonts for premium typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Step 3.3b: Apply the Inter font to the entire app */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Step 3.3c: Style the main app background with a dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1628 100%);
    }
    
    /* Step 3.3d: Style the header section with a gradient card */
    .header-container {
        background: linear-gradient(135deg, #1a2744 0%, #162035 100%);
        border: 1px solid rgba(100, 180, 255, 0.15);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 100, 255, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    /* Step 3.3e: Style for the answer card */
    .answer-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 200, 100, 0.05);
        color: #e2e8f0;
        line-height: 1.7;
    }
    
    /* Step 3.3f: Style for source/chunk cards */
    .source-card {
        background: linear-gradient(135deg, #1e293b 0%, #172033 100%);
        border: 1px solid rgba(96, 165, 250, 0.15);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        color: #cbd5e1;
    }
    
    .source-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 4px 20px rgba(96, 165, 250, 0.1);
        transform: translateY(-2px);
    }
    
    /* Step 3.3g: Severity badge styling */
    .severity-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .severity-low {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .severity-moderate {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .severity-high {
        background: rgba(251, 146, 60, 0.15);
        color: #fb923c;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    
    .severity-critical {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    /* Step 3.3h: Stats card styling */
    .stat-card {
        background: linear-gradient(135deg, #1a2744 0%, #162035 100%);
        border: 1px solid rgba(100, 180, 255, 0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
    
    /* Step 3.3i: Similarity score bar */
    .score-bar-container {
        background: rgba(100, 180, 255, 0.1);
        border-radius: 8px;
        height: 6px;
        margin-top: 8px;
        overflow: hidden;
    }
    
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #60a5fa, #34d399);
        transition: width 0.5s ease;
    }
    
    /* Step 3.3j: Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #0a1020 100%);
    }
    
    /* Step 3.3k: Input field styling */
    .stTextInput input {
        background: #1e293b !important;
        border: 1px solid rgba(100, 180, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Step 3.3l: Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Step 3.3m: Expander styling */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Step 3.3n: Hide Streamlit's default footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Step 3.3o: Spinner/loading animation */
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 3.4: INITIALIZE THE RAG SYSTEM (Cached)
# ============================================================
# @st.cache_resource tells Streamlit to load the RAG system ONCE
# and reuse it across all page refreshes. Without this, the
# knowledge base would reload every time the user interacts
# with the app, making it very slow.
# ============================================================

@st.cache_resource
def initialize_rag():
    """
    STEP 3.4: Load and cache the RAG system.
    
    WHY CACHING?
    ============
    Loading the knowledge base takes a few seconds (reading the .pkl file,
    initializing the Gemini model, etc.). st.cache_resource ensures this
    happens ONLY ONCE, even if the user refreshes the page or interacts
    with widgets. The RAG system stays in memory.
    
    Returns:
        MedRAG instance (or None if the knowledge base doesn't exist)
    """
    try:
        # Step 3.4a: Try to create the RAG system
        return MedRAG()
    except FileNotFoundError as e:
        # Step 3.4b: If knowledge base doesn't exist, return None
        st.error(str(e))
        return None


# ============================================================
# STEP 3.5: HELPER FUNCTION — Get Severity Badge HTML
# ============================================================

def get_severity_badge(severity):
    """
    STEP 3.5: Create a colored HTML badge based on severity level.
    
    This makes severity levels visually distinct:
    - Low      → Green badge
    - Moderate → Yellow badge
    - High     → Orange badge
    - Critical → Red badge
    
    Args:
        severity: String like "Low", "Moderate", "High", "Critical"
    
    Returns:
        HTML string for the colored badge
    """
    
    # Step 3.5a: Map severity to CSS class
    severity_lower = severity.lower()
    css_class = f"severity-{severity_lower}"
    
    # Step 3.5b: Return the HTML badge
    return f'<span class="severity-badge {css_class}">{severity}</span>'


# ============================================================
# STEP 3.6: RENDER THE HEADER
# ============================================================

def render_header():
    """
    STEP 3.6: Display the app header with title and description.
    
    Uses custom HTML/CSS for a premium gradient text effect
    that wouldn't be possible with Streamlit's built-in components.
    """
    
    st.markdown("""
        <div class="header-container">
            <div class="header-title">🏥 MedAgentixAI</div>
            <div class="header-subtitle">
                AI-Powered Medical Knowledge Retrieval System — Ask any medical question and get answers backed by our knowledge base
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# STEP 3.7: RENDER THE SIDEBAR
# ============================================================

def render_sidebar(rag):
    """
    STEP 3.7: Display the sidebar with KB stats and filter options.
    
    The sidebar shows:
    - Knowledge base statistics (total chunks, diseases, etc.)
    - Filter dropdowns (category, severity)
    - Number of results slider
    - Information about the RAG pipeline
    
    Args:
        rag: The initialized MedRAG system
    
    Returns:
        category_filter: Selected category (or "All")
        severity_filter: Selected severity (or "All")
        top_k:           Number of results to retrieve
    """
    
    with st.sidebar:
        # Step 3.7a: Sidebar header
        st.markdown("## ⚙️ Settings & Stats")
        st.markdown("---")
        
        # Step 3.7b: Get knowledge base statistics
        stats = rag.get_stats()
        
        # Step 3.7c: Display stats in a grid layout
        st.markdown("### 📊 Knowledge Base")
        
        # Create two columns for stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['total_chunks']:,}</div>
                    <div class="stat-label">Total Chunks</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['total_diseases']:,}</div>
                    <div class="stat-label">Diseases</div>
                </div>
            """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(stats['categories'])}</div>
                    <div class="stat-label">Categories</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Step 3.7d: Show Gemini status
            gemini_status = "🟢" if stats['has_gemini'] else "🔴"
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{gemini_status}</div>
                    <div class="stat-label">Gemini AI</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Step 3.7e: Filter options
        st.markdown("### 🔍 Search Filters")
        
        # Category filter dropdown
        categories = ["All"] + stats['categories']
        category_filter = st.selectbox(
            "📁 Disease Category",
            categories,
            index=0,
            help="Filter results to a specific medical category"
        )
        
        # Severity filter dropdown
        severities = ["All"] + stats['severities']
        severity_filter = st.selectbox(
            "⚡ Severity Level",
            severities,
            index=0,
            help="Filter results by severity level"
        )
        
        # Number of results slider
        top_k = st.slider(
            "📊 Number of Results",
            min_value=1,
            max_value=10,
            value=5,
            help="How many knowledge chunks to retrieve"
        )
        
        st.markdown("---")
        
        # Step 3.7f: RAG Pipeline information
        st.markdown("### 🔬 How RAG Works")
        st.markdown("""
        1. **Retrieve**: Search knowledge base using TF-IDF + Cosine Similarity
        2. **Augment**: Combine question + relevant medical facts
        3. **Generate**: Gemini AI synthesizes a comprehensive answer
        """)
        
        # Step 3.7g: Show build timestamp
        st.markdown("---")
        st.caption(f"📅 KB Built: {stats['build_timestamp']}")
        
    return category_filter, severity_filter, top_k


# ============================================================
# STEP 3.8: RENDER SEARCH RESULTS
# ============================================================

def render_results(answer, sources):
    """
    STEP 3.8: Display the AI answer and source chunks in the UI.
    
    This function creates two sections:
    1. The AI-generated answer in a styled card
    2. The source knowledge chunks with relevance scores and metadata
    
    Args:
        answer:  The generated answer string
        sources: List of source chunk dicts
    """
    
    # Step 3.8a: Display the AI-generated answer
    st.markdown("### 💡 AI-Generated Answer")
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
    
    # Step 3.8b: Display the source chunks
    st.markdown(f"### 📚 Source Knowledge Chunks ({len(sources)} retrieved)")
    
    # Step 3.8c: Loop through each source and display it
    for i, source in enumerate(sources):
        meta = source['metadata']
        score = source['similarity_score']
        
        # Step 3.8d: Create an expander for each source
        #            Expanders let users click to see more details
        with st.expander(
            f"📄 Source {i+1}: {meta['disease']} — Relevance: {score:.1%}",
            expanded=(i == 0)     # First source is expanded by default
        ):
            # Step 3.8e: Display metadata in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**🏷️ Category:** {meta['category']}")
            with col2:
                st.markdown(f"**⚡ Severity:** {meta['severity']}")
            with col3:
                st.markdown(f"**📊 Relevance:** {score:.1%}")
            
            # Step 3.8f: Display the similarity score as a visual progress bar
            st.progress(min(score, 1.0))
            
            # Step 3.8g: Display the actual chunk text
            st.markdown("**📝 Knowledge Chunk:**")
            st.info(source['chunk'])
            
            # Step 3.8h: Display the source file
            st.caption(f"Source: {meta.get('source', 'unknown')}")


# ============================================================
# STEP 3.9: MAIN APPLICATION FUNCTION
# ============================================================

def main():
    """
    STEP 3.9: Main function that assembles the entire Streamlit app.
    
    This orchestrates:
    1. Header rendering
    2. RAG system initialization
    3. Sidebar with stats and filters
    4. Question input and search
    5. Results display
    
    Streamlit runs this function every time a user interacts with
    the app (clicks a button, types text, etc.). The @st.cache_resource
    decorator ensures the RAG system isn't reloaded each time.
    """
    
    # Step 3.9a: Render the header
    render_header()
    
    # Step 3.9b: Initialize the RAG system (cached — loads only once)
    rag = initialize_rag()
    
    # Step 3.9c: If RAG system failed to load, show an error
    if rag is None:
        st.error("❌ Knowledge base not found! Run `python knowledge_base.py` first.")
        st.code("python knowledge_base.py", language="bash")
        
        # Step 3.9d: Show a helpful info box
        st.info("""
        **Quick Start Guide:**
        1. Make sure your CSV files are in the project directory
        2. Run `python knowledge_base.py` to build the knowledge base
        3. Then run `streamlit run app.py` to start this app
        """)
        return
    
    # Step 3.9e: Render the sidebar and get filter values
    category_filter, severity_filter, top_k = render_sidebar(rag)
    
    # Step 3.9f: Create the question input section
    st.markdown("### 🔎 Ask a Medical Question")
    
    # Step 3.9g: Create two columns — one for input, one for the button
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        # Step 3.9h: Text input for the medical question
        question = st.text_input(
            "Type your medical question here:",
            placeholder="e.g., What are the causes and complications of cardiac syndrome?",
            label_visibility="collapsed"       # Hide the label (we have our own header)
        )
    
    with col_button:
        # Step 3.9i: Search button
        search_clicked = st.button("🔍 Search", use_container_width=True)
    
    # Step 3.9j: Show sample questions for users who don't know what to ask
    if not question:
        st.markdown("#### 💡 Try these sample questions:")
        
        sample_cols = st.columns(3)
        
        sample_questions = [
            "What are cardiac syndromes and their complications?",
            "Tell me about respiratory diseases with high severity",
            "What causes neurological disorders?"
        ]
        
        for i, (col, sample_q) in enumerate(zip(sample_cols, sample_questions)):
            with col:
                if st.button(f"📌 {sample_q}", key=f"sample_{i}", use_container_width=True):
                    # Step 3.9k: If a sample question is clicked, use it
                    question = sample_q
                    search_clicked = True
    
    # Step 3.9l: Process the search when button is clicked OR Enter is pressed
    if (search_clicked or question) and question:
        # Step 3.9m: Show a spinner while processing
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            # Step 3.9n: Record start time for performance tracking
            start_time = time.time()
            
            # Step 3.9o: Run the full RAG pipeline
            #            This calls: retrieve() → generate_answer()
            answer, sources = rag.query(
                question=question,
                top_k=top_k,
                category_filter=category_filter,
                severity_filter=severity_filter
            )
            
            # Step 3.9p: Calculate elapsed time
            elapsed = time.time() - start_time
        
        # Step 3.9q: Show performance metrics
        st.caption(f"⏱️ Query processed in {elapsed:.2f} seconds | 📊 {len(sources)} chunks retrieved")
        
        # Step 3.9r: Display the results
        if sources:
            render_results(answer, sources)
        else:
            # Step 3.9s: No results found
            st.warning("😕 No relevant medical information found for your query.")
            st.info("💡 **Tips:**\n- Try different keywords\n- Remove filters\n- Use medical terminology")
    
    # Step 3.9t: Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem;">
        <p>⚠️ <strong>Disclaimer:</strong> MedAgentixAI is for educational and research purposes only. 
        Do not use this as a substitute for professional medical advice.</p>
        <p>Built with ❤️ using Streamlit, Scikit-learn, and Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# STEP 3.10: SCRIPT ENTRY POINT
# ============================================================
# When Streamlit runs this file (streamlit run app.py),
# it executes the main() function to build the app.
# ============================================================

if __name__ == "__main__":
    main()
