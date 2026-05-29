# ============================================================
# STEP 2: RAG SYSTEM ENGINE — rag_system.py
# ============================================================
# PURPOSE:
#   This is the BRAIN of MedAgentixAI. It implements the full
#   RAG (Retrieval-Augmented Generation) pipeline:
#   
#   1. RETRIEVE: Find the most relevant knowledge chunks for a question
#   2. AUGMENT:  Combine the question + retrieved chunks into a prompt
#   3. GENERATE: Use Google Gemini AI to generate a medical answer
#
# WHAT IS RAG? (Detailed Explanation)
# =========================================
#   RAG = Retrieval-Augmented Generation
#   
#   Traditional AI (like ChatGPT) answers from its training data,
#   which can be outdated or hallucinate (make things up).
#   
#   RAG solves this by:
#   1. FIRST searching a knowledge base for relevant information
#   2. THEN feeding that information to an AI to generate an answer
#   
#   This means the AI always has REAL facts to work with, reducing
#   hallucination and ensuring accurate, source-backed answers.
#
# RAG PIPELINE IN THIS FILE:
#   User Question → TF-IDF Vectorize → Cosine Similarity Search →
#   Top-K Chunks Retrieved → Build Augmented Prompt → Gemini AI →
#   Final Answer with Sources
# ============================================================


# ============================================================
# STEP 2.1: IMPORT ALL REQUIRED LIBRARIES
# ============================================================

import pickle                                             # Step 2.1a: For loading the saved knowledge base (.pkl file)

from sklearn.metrics.pairwise import cosine_similarity    # Step 2.1b: For calculating similarity between question and chunks
                                                          #            Cosine similarity measures the angle between two vectors
                                                          #            1.0 = identical, 0.0 = completely different

import numpy as np                                        # Step 2.1c: For numerical operations on arrays/matrices

import google.generativeai as genai                       # Step 2.1d: Google's Gemini AI SDK for generating answers

from dotenv import load_dotenv                            # Step 2.1e: For loading the API key from .env file securely

import os                                                 # Step 2.1f: For accessing environment variables

import sys
# Reconfigure stdout to UTF-8 to handle clinical unicode symbols/emojis safely on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================
# STEP 2.2: LOAD ENVIRONMENT VARIABLES & CONFIGURE GEMINI
# ============================================================

# Step 2.2a: Load the .env file which contains our GEMINI_API_KEY
#            load_dotenv() reads the .env file and makes its variables
#            available via os.getenv()
load_dotenv()


# ============================================================
# STEP 2.3: THE MedRAG CLASS — Our RAG System
# ============================================================
# We use a CLASS to organize all RAG functionality together.
# A class is like a blueprint — it defines what our RAG system
# CAN DO (methods) and what it KNOWS (attributes).
# ============================================================

class MedRAG:
    """
    MedRAG — Medical Retrieval-Augmented Generation System.
    
    This class handles the entire RAG pipeline:
    1. Loading the pre-built knowledge base
    2. Searching for relevant medical information
    3. Generating AI-powered answers using Google Gemini
    
    Usage:
        rag = MedRAG()                           # Create the RAG system
        answer, sources = rag.query("What is cardiac syndrome?")  # Ask a question
    """
    
    # ========================================================
    # STEP 2.3a: CONSTRUCTOR — Initialize the RAG System
    # ========================================================
    
    def __init__(self, knowledge_base_path="knowledge_base.pkl"):
        """
        STEP 2.3a: Initialize the RAG system by loading the knowledge base
        and configuring the Gemini AI model.
        
        This runs automatically when you create a MedRAG object:
            rag = MedRAG()  ← this calls __init__
        
        Args:
            knowledge_base_path: Path to the .pkl file built by knowledge_base.py
        """
        
        print("🏥 Initializing MedAgentixAI RAG System...")
        
        # Step 2.3a-i: Load the knowledge base from disk
        #              This was built by knowledge_base.py
        self.kb = self._load_knowledge_base(knowledge_base_path)
        
        # Step 2.3a-ii: Configure the Gemini AI model
        self._configure_gemini()
        
        print("✅ RAG System initialized successfully!\n")
    
    
    # ========================================================
    # STEP 2.3b: LOAD KNOWLEDGE BASE FROM DISK
    # ========================================================
    
    def _load_knowledge_base(self, path):
        """
        STEP 2.3b: Load the pre-built knowledge base from a pickle file.
        
        The underscore prefix (_) means this is a PRIVATE method —
        it's only used internally by the class, not called by users.
        
        Args:
            path: File path to the .pkl knowledge base
        
        Returns:
            Dictionary containing vectors, chunks, metadata, etc.
        """
        
        # Step 2.3b-i: Check if the knowledge base file exists
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Knowledge base not found at '{path}'!\n"
                f"   Run 'python knowledge_base.py' first to build it."
            )
        
        # Step 2.3b-ii: Open and load the pickle file
        #               'rb' means "read binary"
        with open(path, 'rb') as f:
            kb = pickle.load(f)
        
        # Step 2.3b-iii: Print what was loaded
        print(f"   📂 Knowledge base loaded: {len(kb['chunks'])} chunks")
        print(f"   📊 Categories: {kb['categories']}")
        print(f"   📊 Built at: {kb['build_timestamp']}")
        
        return kb
    
    
    # ========================================================
    # STEP 2.3c: CONFIGURE GOOGLE GEMINI AI
    # ========================================================
    
    def _configure_gemini(self):
        """
        STEP 2.3c: Set up the Google Gemini AI model for answer generation.
        
        WHAT IS GEMINI?
        ================
        Gemini is Google's latest AI model (like ChatGPT but by Google).
        We use it to generate natural language answers from the retrieved
        medical knowledge chunks. The free tier allows plenty of requests.
        
        HOW WE USE IT IN RAG:
        ================
        We DON'T ask Gemini to answer from its own knowledge.
        Instead, we give it the relevant chunks from our knowledge base
        and ask it to synthesize an answer from THAT information only.
        This keeps answers accurate and grounded in our data.
        """
        
        # Step 2.3c-i: Get the API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Step 2.3c-ii: Check if the API key was found
        if not api_key or api_key == "paste_your_gemini_api_key_here":
            print("   ⚠️  WARNING: Gemini API key not found!")
            print("   ⚠️  Add your key to the .env file: GEMINI_API_KEY=your_key_here")
            print("   ⚠️  Get a free key at: https://aistudio.google.com/apikey")
            print("   ⚠️  The system will work but use template-based answers instead.\n")
            self.gemini_model = None
            return
        
        # Step 2.3c-iii: Configure the Gemini SDK with our API key
        genai.configure(api_key=api_key)
        
        # Step 2.3c-iv: Create a Gemini model instance
        #               'gemini-2.0-flash' is lightweight and free-tier friendly
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        print("   🤖 Gemini AI model configured (gemini-2.0-flash)")
    
    
    # ========================================================
    # STEP 2.4: RETRIEVE — Find Relevant Knowledge Chunks
    # ========================================================
    
    def retrieve(self, question, top_k=5, category_filter=None, severity_filter=None):
        """
        STEP 2.4: RETRIEVAL — The "R" in RAG.
        
        This is the SEARCH ENGINE of our RAG system. Given a user's question,
        it finds the top-K most relevant knowledge chunks using cosine similarity.
        
        HOW COSINE SIMILARITY WORKS:
        ============================
        Imagine two arrows (vectors) in space:
        - If they point in the SAME direction → similarity = 1.0 (identical meaning)
        - If they are PERPENDICULAR → similarity = 0.0 (unrelated)
        - If they point in OPPOSITE directions → similarity = -1.0 (opposite meaning)
        
        We convert both the question AND each chunk into TF-IDF vectors,
        then calculate the cosine of the angle between them.
        
        EXAMPLE:
        ========
        Question: "What are cardiac diseases?"
        Chunk 1: "Cardiac Syndrome 11 affects cardiovascular systems..." → similarity: 0.82
        Chunk 2: "Respiratory Syndrome 5 affects breathing systems..."   → similarity: 0.15
        → Chunk 1 is much more relevant!
        
        Args:
            question:         The user's medical question (string)
            top_k:            How many top results to return (default: 5)
            category_filter:  Optional filter by disease category (e.g., "Cardiovascular")
            severity_filter:  Optional filter by severity level (e.g., "High")
        
        Returns:
            results: List of dicts, each containing:
                     - 'chunk': The text of the relevant chunk
                     - 'metadata': Disease name, category, severity, etc.
                     - 'similarity_score': How relevant it is (0.0 to 1.0)
        """
        
        # Step 2.4a: Convert the user's question into a TF-IDF vector
        #            We use the SAME vectorizer that was used to build the KB
        #            transform() converts text to vector using the learned vocabulary
        #            (fit_transform was used during building; now we only transform)
        question_vector = self.kb['vectorizer'].transform([question])
        
        # Step 2.4b: Calculate cosine similarity between the question and ALL chunks
        #            This gives us a score for each chunk (how similar to the question)
        #            Result shape: (1, num_chunks) — one score per chunk
        similarity_scores = cosine_similarity(question_vector, self.kb['tfidf_matrix'])
        
        # Step 2.4c: Flatten the array from 2D to 1D for easier processing
        #            [[0.82, 0.15, 0.45, ...]] → [0.82, 0.15, 0.45, ...]
        scores = similarity_scores.flatten()
        
        # Step 2.4d: Apply category filter if specified
        #            This lets users narrow results to specific disease categories
        if category_filter and category_filter != "All":
            for i, meta in enumerate(self.kb['metadata']):
                if meta['category'].lower() != category_filter.lower():
                    scores[i] = 0.0   # Zero out scores for non-matching categories
        
        # Step 2.4e: Apply severity filter if specified
        if severity_filter and severity_filter != "All":
            for i, meta in enumerate(self.kb['metadata']):
                if meta['severity'].lower() != severity_filter.lower():
                    scores[i] = 0.0   # Zero out scores for non-matching severities
        
        # Step 2.4f: Get the indices of the top-K highest scores
        #            argsort() sorts indices by score (ascending), so we take the last K
        #            [::-1] reverses to get descending order (highest first)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Step 2.4g: Build the results list with chunks, metadata, and scores
        results = []
        for idx in top_indices:
            # Step 2.4h: Only include results with a meaningful similarity score
            if scores[idx] > 0.0:
                results.append({
                    'chunk': self.kb['chunks'][idx],                  # The actual text
                    'metadata': self.kb['metadata'][idx],             # Disease info
                    'similarity_score': round(float(scores[idx]), 4)  # Relevance score
                })
        
        return results
    
    
    # ========================================================
    # STEP 2.5: GENERATE — Create an Answer Using Gemini AI
    # ========================================================
    
    def generate_answer(self, question, retrieved_chunks):
        """
        STEP 2.5: GENERATION — The "G" in RAG.
        
        This takes the user's question and the retrieved knowledge chunks,
        combines them into a prompt, and sends it to Google Gemini AI
        to generate a comprehensive medical answer.
        
        THE KEY INSIGHT OF RAG:
        ======================
        We don't ask the AI to answer from its own memory.
        We GIVE it specific medical facts and ask it to synthesize
        an answer from THOSE facts only. This is like giving a
        student a textbook and asking them to answer from it.
        
        Args:
            question:          The user's medical question
            retrieved_chunks:  List of relevant chunks from the retrieve() step
        
        Returns:
            answer: A generated text answer (string)
        """
        
        # Step 2.5a: Build the CONTEXT string from retrieved chunks
        #            This is the "knowledge" we feed to the AI
        context = ""
        for i, chunk_data in enumerate(retrieved_chunks):
            context += f"\n--- Source {i+1} (Relevance: {chunk_data['similarity_score']}) ---\n"
            context += chunk_data['chunk'] + "\n"
        
        # Step 2.5b: Check if Gemini is available
        #            If not (no API key), fall back to template-based answer
        if self.gemini_model is None:
            return self._template_answer(question, retrieved_chunks)
        
        # Step 2.5c: Build the AUGMENTED PROMPT
        #            This prompt tells Gemini exactly what to do:
        #            - You are a medical AI assistant
        #            - Here is the knowledge base context
        #            - Answer THIS question using THAT context
        #            - Be professional and structured
        prompt = f"""You are MedAgentixAI, a professional medical knowledge assistant.

IMPORTANT INSTRUCTIONS:
1. Answer the question ONLY using the medical knowledge provided below.
2. If the information is not in the provided context, say "I don't have enough information in my knowledge base to answer this."
3. Be professional, clear, and structured in your response.
4. Include relevant details like severity, cause, management, and complications when available.
5. Add a disclaimer that this is for educational purposes only, not medical advice.

=== RETRIEVED MEDICAL KNOWLEDGE ===
{context}

=== USER QUESTION ===
{question}

=== YOUR ANSWER ===
Provide a comprehensive, well-structured answer:"""
        
        # Step 2.5d: Send the prompt to Gemini AI and get the response
        try:
            # Step 2.5d-i: Call the Gemini API
            #              generate_content() sends our prompt and returns AI-generated text
            response = self.gemini_model.generate_content(prompt)
            
            # Step 2.5d-ii: Extract the text from the response object
            answer = response.text
            
            return answer
        
        except Exception as e:
            # Step 2.5d-iii: If Gemini fails (network error, rate limit, etc.),
            #                fall back to template-based answer
            print(f"   ⚠️  Gemini API error: {e}")
            print(f"   ⚠️  Falling back to template-based answer...")
            return self._template_answer(question, retrieved_chunks)
    
    
    # ========================================================
    # STEP 2.6: TEMPLATE-BASED FALLBACK ANSWER
    # ========================================================
    
    def _template_answer(self, question, retrieved_chunks):
        """
        STEP 2.6: Fallback answer generation when Gemini AI is unavailable.
        
        This creates a structured answer using the retrieved chunks directly,
        WITHOUT using any AI model. It's a simple but effective fallback.
        
        Args:
            question:          The user's question
            retrieved_chunks:  Retrieved knowledge chunks
        
        Returns:
            answer: A formatted text answer (string)
        """
        
        # Step 2.6a: Start building the answer string
        answer = f"📋 **Medical Knowledge Results for:** *{question}*\n\n"
        
        # Step 2.6b: Check if we found any relevant chunks
        if not retrieved_chunks:
            answer += "❌ No relevant medical information found in the knowledge base.\n"
            answer += "Please try rephrasing your question or use different keywords.\n"
            return answer
        
        # Step 2.6c: Format each retrieved chunk into the answer
        for i, chunk_data in enumerate(retrieved_chunks):
            meta = chunk_data['metadata']
            score = chunk_data['similarity_score']
            
            answer += f"### 🔬 Finding {i+1} (Relevance: {score:.1%})\n"
            answer += f"- **Disease:** {meta['disease']}\n"
            answer += f"- **Category:** {meta['category']}\n"
            answer += f"- **Severity:** {meta['severity']}\n"
            answer += f"- **Details:** {chunk_data['chunk'][:300]}...\n\n"
        
        # Step 2.6d: Add a disclaimer
        answer += "\n---\n⚠️ *This information is for educational purposes only. "
        answer += "Please consult a healthcare professional for medical advice.*"
        
        return answer
    
    
    # ========================================================
    # STEP 2.7: QUERY — The Main Entry Point (Combines R + A + G)
    # ========================================================
    
    def query(self, question, top_k=5, category_filter=None, severity_filter=None):
        """
        STEP 2.7: The MAIN method that runs the full RAG pipeline.
        
        This is the method users should call. It combines:
        1. RETRIEVE: Search the knowledge base
        2. AUGMENT:  Build the prompt with context
        3. GENERATE: Create an answer with Gemini AI
        
        Args:
            question:         The user's medical question
            top_k:            Number of chunks to retrieve (default: 5)
            category_filter:  Optional category filter
            severity_filter:  Optional severity filter
        
        Returns:
            answer:           The generated answer (string)
            retrieved_chunks: The source chunks used (list of dicts)
        """
        
        # Step 2.7a: RETRIEVE — Find relevant knowledge chunks
        retrieved_chunks = self.retrieve(
            question, 
            top_k=top_k,
            category_filter=category_filter,
            severity_filter=severity_filter
        )
        
        # Step 2.7b: GENERATE — Create an answer from the chunks
        answer = self.generate_answer(question, retrieved_chunks)
        
        # Step 2.7c: Return both the answer and the source chunks
        #            (so the UI can display sources separately)
        return answer, retrieved_chunks
    
    
    # ========================================================
    # STEP 2.8: GET KNOWLEDGE BASE STATISTICS
    # ========================================================
    
    def get_stats(self):
        """
        STEP 2.8: Return statistics about the knowledge base.
        
        This is used by the Streamlit sidebar to display KB info.
        
        Returns:
            Dictionary with statistics about the knowledge base
        """
        
        return {
            'total_chunks': len(self.kb['chunks']),
            'total_diseases': self.kb['total_diseases'],
            'categories': self.kb['categories'],
            'severities': self.kb['severities'],
            'build_timestamp': self.kb['build_timestamp'],
            'has_gemini': self.gemini_model is not None
        }


# ============================================================
# STEP 2.9: TEST THE RAG SYSTEM (Standalone Execution)
# ============================================================

if __name__ == "__main__":
    """
    STEP 2.9: Test the RAG system when running this file directly.
    
    Run: python rag_system.py
    
    This will:
    1. Load the knowledge base
    2. Ask a sample question
    3. Print the answer and sources
    """
    
    print("\n" + "🏥" * 30)
    print("   MedAgentixAI — RAG System Test")
    print("🏥" * 30 + "\n")
    
    # Step 2.9a: Create the RAG system (loads KB + configures Gemini)
    rag = MedRAG()
    
    # Step 2.9b: Test with a sample question
    test_questions = [
        "What are the causes and complications of cardiac syndrome?",
        "Tell me about respiratory diseases with high severity",
        "What is the management for neurological conditions?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        
        # Step 2.9c: Run the full RAG pipeline
        answer, sources = rag.query(question, top_k=3)
        
        # Step 2.9d: Print the answer
        print(f"\n💡 Answer:\n{answer}")
        
        # Step 2.9e: Print the sources
        print(f"\n📚 Sources used: {len(sources)}")
        for i, src in enumerate(sources):
            print(f"   {i+1}. {src['metadata']['disease']} (Score: {src['similarity_score']})")
        
        print()
