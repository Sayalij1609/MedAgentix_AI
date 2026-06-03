# ============================================================
# STEP 1: KNOWLEDGE BASE BUILDER — knowledge_base.py
# ============================================================
# PURPOSE:
#   This script reads the medical CSV datasets, processes them,
#   creates TF-IDF vector embeddings, and saves everything as a
#   reusable knowledge base file (.pkl). This is the FOUNDATION
#   of our RAG system — without this, there's nothing to retrieve!
#
# HOW IT WORKS (Big Picture):
#   CSV Files → Clean Data → Text Chunks → TF-IDF Vectors → Save to Disk
#
# WHAT IS TF-IDF?
#   TF-IDF (Term Frequency - Inverse Document Frequency) converts text
#   into numbers (vectors). It measures how important a word is to a
#   document. Words that appear often in ONE document but rarely in
#   OTHER documents get higher scores. This lets us mathematically
#   compare how similar two pieces of text are.
#
# RUN THIS SCRIPT:
#   python knowledge_base.py
# ============================================================


# ============================================================
# STEP 1.1: IMPORT ALL REQUIRED LIBRARIES
# ============================================================

import pandas as pd                          # Step 1.1a: pandas is used to load and manipulate CSV files
                                             #            Think of it as "Excel for Python"

from sklearn.feature_extraction.text import TfidfVectorizer  # Step 1.1b: Converts text into numerical TF-IDF vectors
                                                              #            This is the core of our search engine

import pickle                                # Step 1.1c: pickle lets us save Python objects to disk
                                             #            So we don't have to rebuild the KB every time

import os                                    # Step 1.1d: os provides functions to work with file paths
                                             #            and check if files exist

import time                                  # Step 1.1e: time is used to measure how long each step takes


# ============================================================
# STEP 1.2: DEFINE FILE PATHS
# ============================================================
# We define all file paths at the top so they're easy to find
# and change if the files are moved to a different location.
# ============================================================

# Step 1.2a: Path to the raw medical knowledge dataset (5,001 diseases)
#            This CSV has columns: Disease, Description, Cause, Category,
#            Severity, Disease_Progression, Common_Complications, Prevalence, Primary_Management
MEDICAL_DATASET_PATH = "Medical Knowledge Dataset.csv"

# Step 1.2b: Path to the pre-chunked knowledge dataset (~3,000 text chunks)
#            This CSV has columns: disease, category, severity, text_chunk, chunk_length
#            Each text_chunk is a ready-to-use paragraph about a disease
KNOWLEDGE_CHUNKS_PATH = "knowledge_chunks.csv"

# Step 1.2c: Path to the 1000-row medical dataset with symptoms & advice
#            This CSV has columns: id, disease_name, category, severity,
#            target_demographic, symptoms, description, medical_advice
MEDICAL_1000_PATH = "medical_dataset_1000_rows.csv"

# Step 1.2d: Path where the final knowledge base will be saved
#            This .pkl file contains all vectors and metadata
KNOWLEDGE_BASE_OUTPUT = "knowledge_base.pkl"


# ============================================================
# STEP 1.3: LOAD THE CSV DATASETS
# ============================================================

def load_datasets():
    """
    STEP 1.3: Load all three CSV files into pandas DataFrames.
    
    A DataFrame is like a table (rows and columns) in Python.
    We load all datasets because:
    - Medical Knowledge Dataset: has structured fields (cause, severity, etc.)
    - Knowledge Chunks: has pre-formatted text paragraphs for retrieval
    - Medical 1000 Rows: has symptoms, demographics, and medical advice
    
    Returns:
        medical_df:    DataFrame with the raw medical dataset
        chunks_df:     DataFrame with the knowledge chunks
        medical_1000:  DataFrame with the 1000-row medical dataset
    """
    
    # Step 1.3a: Print a status message so the user knows what's happening
    print("=" * 60)
    print("📂 STEP 1.3: Loading CSV datasets...")
    print("=" * 60)
    
    # Step 1.3b: Load the Medical Knowledge Dataset CSV
    #            encoding='utf-8' handles special characters properly
    medical_df = pd.read_csv(MEDICAL_DATASET_PATH, encoding='utf-8')
    
    # Step 1.3c: Print how many rows and columns were loaded
    print(f"   ✅ Medical Knowledge Dataset loaded: {medical_df.shape[0]} rows × {medical_df.shape[1]} columns")
    
    # Step 1.3d: Load the Knowledge Chunks CSV
    chunks_df = pd.read_csv(KNOWLEDGE_CHUNKS_PATH, encoding='utf-8')
    
    # Step 1.3e: Print how many chunks were loaded
    print(f"   ✅ Knowledge Chunks loaded: {chunks_df.shape[0]} rows × {chunks_df.shape[1]} columns")
    
    # Step 1.3f: Load the 1000-row Medical Dataset
    medical_1000 = pd.read_csv(MEDICAL_1000_PATH, encoding='utf-8')
    
    # Step 1.3g: Print how many rows were loaded
    print(f"   ✅ Medical Dataset (1000 rows) loaded: {medical_1000.shape[0]} rows × {medical_1000.shape[1]} columns")
    
    # Step 1.3h: Return all three DataFrames for further processing
    return medical_df, chunks_df, medical_1000


# ============================================================
# STEP 1.4: CLEAN AND PREPROCESS THE DATA
# ============================================================

def clean_data(medical_df, chunks_df, medical_1000):
    """
    STEP 1.4: Clean the data to remove problems like:
    - Missing values (empty cells)
    - Duplicate rows (same disease listed twice)
    - Extra whitespace in text
    
    Clean data = better search results!
    
    Args:
        medical_df:   Raw medical dataset DataFrame
        chunks_df:    Raw knowledge chunks DataFrame
        medical_1000: Raw 1000-row medical dataset DataFrame
    
    Returns:
        medical_df:   Cleaned medical dataset
        chunks_df:    Cleaned knowledge chunks
        medical_1000: Cleaned 1000-row medical dataset
    """
    
    print("\n" + "=" * 60)
    print("🧹 STEP 1.4: Cleaning and preprocessing data...")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Step 1.4a: Handle Missing Values
    # --------------------------------------------------------
    # fillna('Unknown') replaces any empty cell with the word "Unknown"
    # This prevents errors when we try to process text later
    
    # Count missing values before cleaning
    medical_missing = medical_df.isnull().sum().sum()     # Total missing cells in medical dataset
    chunks_missing = chunks_df.isnull().sum().sum()       # Total missing cells in chunks dataset
    m1000_missing = medical_1000.isnull().sum().sum()     # Total missing cells in 1000-row dataset
    
    print(f"   📊 Missing values found — Medical: {medical_missing}, Chunks: {chunks_missing}, Medical-1000: {m1000_missing}")
    
    # Fill missing values with 'Unknown' so no cell is empty
    medical_df = medical_df.fillna('Unknown')
    chunks_df = chunks_df.fillna('Unknown')
    medical_1000 = medical_1000.fillna('Unknown')
    
    print(f"   ✅ Missing values filled with 'Unknown'")
    
    # --------------------------------------------------------
    # Step 1.4b: Remove Duplicate Rows
    # --------------------------------------------------------
    # drop_duplicates() removes rows that are exact copies
    # This prevents the same disease from appearing twice in results
    
    medical_before = len(medical_df)                       # Count rows before removing duplicates
    medical_df = medical_df.drop_duplicates()              # Remove duplicate rows
    medical_after = len(medical_df)                        # Count rows after
    
    chunks_before = len(chunks_df)
    chunks_df = chunks_df.drop_duplicates(subset=['text_chunk'])  # Remove chunks with identical text
    chunks_after = len(chunks_df)
    
    m1000_before = len(medical_1000)
    medical_1000 = medical_1000.drop_duplicates(subset=['description'])  # Remove rows with identical descriptions
    m1000_after = len(medical_1000)
    
    print(f"   🗑️  Duplicates removed — Medical: {medical_before - medical_after}, Chunks: {chunks_before - chunks_after}, Medical-1000: {m1000_before - m1000_after}")
    
    # --------------------------------------------------------
    # Step 1.4c: Strip Extra Whitespace from Text Columns
    # --------------------------------------------------------
    # .str.strip() removes leading/trailing spaces from text
    # "  hello  " becomes "hello"
    
    # Clean text columns in chunks dataset
    if 'text_chunk' in chunks_df.columns:
        chunks_df['text_chunk'] = chunks_df['text_chunk'].astype(str).str.strip()
    
    # Clean the Disease column in medical dataset
    if 'Disease' in medical_df.columns:
        medical_df['Disease'] = medical_df['Disease'].astype(str).str.strip()
    
    # Clean text columns in 1000-row dataset
    if 'disease_name' in medical_1000.columns:
        medical_1000['disease_name'] = medical_1000['disease_name'].astype(str).str.strip()
    if 'description' in medical_1000.columns:
        medical_1000['description'] = medical_1000['description'].astype(str).str.strip()
    
    print(f"   ✅ Whitespace stripped from text columns")
    print(f"   📊 Final counts — Medical: {len(medical_df)} rows, Chunks: {len(chunks_df)} rows, Medical-1000: {len(medical_1000)} rows")
    
    return medical_df, chunks_df, medical_1000


# ============================================================
# STEP 1.5: CREATE ENRICHED TEXT CHUNKS
# ============================================================

def create_enriched_chunks(medical_df, chunks_df, medical_1000):
    """
    STEP 1.5: Combine data from all three datasets to create RICHER text chunks.
    
    WHY?
    The knowledge_chunks.csv already has text paragraphs, but we can make
    them even better by adding structured data from the Medical Knowledge
    Dataset. The 1000-row dataset adds symptoms, demographics, and medical
    advice. More information = better answers!
    
    We also create additional chunks from the medical datasets for diseases
    that might not be in the chunks file.
    
    Args:
        medical_df:   Cleaned medical dataset
        chunks_df:    Cleaned knowledge chunks
        medical_1000: Cleaned 1000-row medical dataset
    
    Returns:
        all_chunks:   List of text strings (each chunk is a paragraph)
        all_metadata: List of dicts with metadata for each chunk
    """
    
    print("\n" + "=" * 60)
    print("🔧 STEP 1.5: Creating enriched knowledge chunks...")
    print("=" * 60)
    
    all_chunks = []      # Step 1.5a: This list will hold ALL text chunks (strings)
    all_metadata = []    # Step 1.5b: This list will hold metadata dicts for each chunk
                         #            Metadata = extra info like disease name, category, severity
    
    # --------------------------------------------------------
    # Step 1.5c: Process chunks from knowledge_chunks.csv
    # --------------------------------------------------------
    # These chunks are already formatted as nice text paragraphs
    # We just need to extract them and store their metadata
    
    print(f"   📝 Processing {len(chunks_df)} knowledge chunks...")
    
    for index, row in chunks_df.iterrows():
        # iterrows() loops through each row of the DataFrame
        # 'index' is the row number, 'row' is the data in that row
        
        # Step 1.5d: Get the text chunk (the main content for search)
        chunk_text = str(row.get('text_chunk', ''))
        
        # Step 1.5e: Skip empty chunks (they won't help with search)
        if len(chunk_text.strip()) < 10:
            continue
        
        # Step 1.5f: Add the chunk text to our list
        all_chunks.append(chunk_text)
        
        # Step 1.5g: Store metadata so we can display it alongside results
        all_metadata.append({
            'disease': str(row.get('disease', 'Unknown')),
            'category': str(row.get('category', 'Unknown')),
            'severity': str(row.get('severity', 'Unknown')),
            'source': 'knowledge_chunks',                      # Track where this chunk came from
            'chunk_length': int(row.get('chunk_length', 0))
        })
    
    print(f"   ✅ Added {len(all_chunks)} chunks from knowledge_chunks.csv")
    
    # --------------------------------------------------------
    # Step 1.5h: Create additional chunks from Medical Knowledge Dataset
    # --------------------------------------------------------
    # For each disease in the medical dataset, we create a structured
    # text paragraph combining ALL its fields. This gives the RAG
    # system more information to work with.
    
    print(f"   📝 Creating enriched chunks from Medical Knowledge Dataset...")
    
    enriched_count = 0
    
    for index, row in medical_df.iterrows():
        # Step 1.5i: Build a rich text paragraph from ALL columns
        #            We format it as a structured description
        enriched_text = (
            f"Disease: {row.get('Disease', 'Unknown')}. "
            f"Description: {row.get('Description', 'Unknown')}. "
            f"Cause: {row.get('Cause', 'Unknown')}. "
            f"Category: {row.get('Category', 'Unknown')}. "
            f"Severity: {row.get('Severity', 'Unknown')}. "
            f"Disease Progression: {row.get('Disease_Progression', 'Unknown')}. "
            f"Common Complications: {row.get('Common_Complications', 'Unknown')}. "
            f"Prevalence: {row.get('Prevalence', 'Unknown')}. "
            f"Primary Management: {row.get('Primary_Management', 'Unknown')}."
        )
        
        # Step 1.5j: Add the enriched chunk to our lists
        all_chunks.append(enriched_text)
        
        all_metadata.append({
            'disease': str(row.get('Disease', 'Unknown')),
            'category': str(row.get('Category', 'Unknown')),
            'severity': str(row.get('Severity', 'Unknown')),
            'source': 'medical_dataset',                        # Track the source
            'chunk_length': len(enriched_text)
        })
        
        enriched_count += 1
    
    print(f"   ✅ Created {enriched_count} enriched chunks from Medical Knowledge Dataset")
    
    # --------------------------------------------------------
    # Step 1.5k: Create chunks from the 1000-row Medical Dataset
    # --------------------------------------------------------
    # This dataset has symptoms, target demographics, and medical advice
    # which adds valuable clinical detail to the knowledge base.
    
    print(f"   📝 Creating chunks from Medical Dataset (1000 rows)...")
    
    m1000_count = 0
    
    for index, row in medical_1000.iterrows():
        # Step 1.5l: Build a rich text paragraph from all columns
        enriched_text = (
            f"Disease: {row.get('disease_name', 'Unknown')}. "
            f"Category: {row.get('category', 'Unknown')}. "
            f"Severity: {row.get('severity', 'Unknown')}. "
            f"Target Demographic: {row.get('target_demographic', 'Unknown')}. "
            f"Symptoms: {row.get('symptoms', 'Unknown')}. "
            f"Description: {row.get('description', 'Unknown')}. "
            f"Medical Advice: {row.get('medical_advice', 'Unknown')}."
        )
        
        # Step 1.5m: Skip very short chunks
        if len(enriched_text.strip()) < 20:
            continue
        
        # Step 1.5n: Add the chunk and its metadata
        all_chunks.append(enriched_text)
        
        all_metadata.append({
            'disease': str(row.get('disease_name', 'Unknown')),
            'category': str(row.get('category', 'Unknown')),
            'severity': str(row.get('severity', 'Unknown')),
            'source': 'medical_1000',                            # Track the source
            'chunk_length': len(enriched_text)
        })
        
        m1000_count += 1
    
    print(f"   ✅ Created {m1000_count} chunks from Medical Dataset (1000 rows)")
    print(f"   📊 Total knowledge chunks: {len(all_chunks)}")
    
    return all_chunks, all_metadata


# ============================================================
# STEP 1.6: BUILD TF-IDF VECTOR INDEX
# ============================================================

def build_tfidf_index(all_chunks):
    """
    STEP 1.6: Convert all text chunks into TF-IDF vectors.
    
    WHAT IS TF-IDF? (Detailed Explanation)
    =========================================
    TF-IDF stands for Term Frequency - Inverse Document Frequency.
    
    It works in two parts:
    
    1. TF (Term Frequency): How often a word appears in a SINGLE document.
       → If "cardiac" appears 5 times in a chunk, it has high TF.
    
    2. IDF (Inverse Document Frequency): How RARE a word is across ALL documents.
       → If "cardiac" appears in only 10 out of 8000 chunks, it has high IDF.
       → If "the" appears in ALL 8000 chunks, it has very low IDF.
    
    TF × IDF = Final Score
       → Words that are FREQUENT in one chunk but RARE overall get high scores.
       → This helps us find the most RELEVANT chunks for a question.
    
    WHY TF-IDF FOR RAG?
    =========================================
    - It's FAST (no GPU needed, runs on any laptop)
    - It works well for keyword-based medical queries
    - It's TRANSPARENT (you can see which words matched)
    - It handles large datasets efficiently
    
    Args:
        all_chunks: List of text strings to vectorize
    
    Returns:
        vectorizer:  The fitted TfidfVectorizer object (needed to vectorize new queries)
        tfidf_matrix: The TF-IDF matrix (each row = one chunk's vector representation)
    """
    
    print("\n" + "=" * 60)
    print("🧠 STEP 1.6: Building TF-IDF vector index...")
    print("=" * 60)
    
    # Step 1.6a: Record the start time to measure performance
    start_time = time.time()
    
    # Step 1.6b: Create the TF-IDF Vectorizer with specific settings
    vectorizer = TfidfVectorizer(
        max_features=10000,         # Step 1.6c: Only keep the top 10,000 most important words
                                    #            This reduces memory usage and speeds up search
        
        stop_words='english',       # Step 1.6d: Remove common English words like "the", "is", "and"
                                    #            These words don't help distinguish between documents
        
        ngram_range=(1, 2),         # Step 1.6e: Use both single words AND two-word phrases
                                    #            (1,2) means: "cardiac" AND "cardiac syndrome"
                                    #            This captures important medical terms that are multi-word
        
        min_df=2,                   # Step 1.6f: A word must appear in at least 2 chunks to be included
                                    #            This removes ultra-rare typos and noise
        
        max_df=0.95,                # Step 1.6g: A word can appear in at most 95% of chunks
                                    #            This removes words so common they don't help with search
        
        sublinear_tf=True           # Step 1.6h: Use logarithmic TF scaling (1 + log(tf))
                                    #            This prevents very long documents from dominating results
                                    #            A word appearing 100 times isn't 100x more important than 1 time
    )
    
    print(f"   ⚙️  Vectorizer configured: max_features=10000, ngram_range=(1,2)")
    
    # Step 1.6i: FIT and TRANSFORM the text chunks into TF-IDF vectors
    #            fit_transform() does TWO things:
    #            1. FIT: Learn the vocabulary from all chunks (which words exist)
    #            2. TRANSFORM: Convert each chunk into a numerical vector
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    
    # Step 1.6j: Calculate and print performance metrics
    elapsed = time.time() - start_time
    vocab_size = len(vectorizer.vocabulary_)
    
    print(f"   ✅ TF-IDF matrix built successfully!")
    print(f"   📊 Matrix shape: {tfidf_matrix.shape} (chunks × vocabulary)")
    print(f"   📊 Vocabulary size: {vocab_size} unique terms")
    print(f"   ⏱️  Time taken: {elapsed:.2f} seconds")
    
    return vectorizer, tfidf_matrix


# ============================================================
# STEP 1.7: SAVE KNOWLEDGE BASE TO DISK
# ============================================================

def save_knowledge_base(vectorizer, tfidf_matrix, all_chunks, all_metadata, medical_df):
    """
    STEP 1.7: Save everything into a single .pkl (pickle) file.
    
    WHAT IS PICKLE?
    =========================================
    Pickle is Python's built-in serialization format. It converts
    Python objects (lists, dicts, matrices) into a binary file that
    can be loaded back later. Think of it as "saving your game."
    
    WHY DO WE SAVE?
    =========================================
    Building the TF-IDF index takes time. By saving it, we only need
    to build it ONCE. The RAG system and Streamlit app can load the
    pre-built knowledge base instantly.
    
    Args:
        vectorizer:   The fitted TF-IDF vectorizer (needed for new queries)
        tfidf_matrix: The TF-IDF matrix of all chunks
        all_chunks:   List of text strings
        all_metadata: List of metadata dicts
        medical_df:   Original medical dataset for statistics
    """
    
    print("\n" + "=" * 60)
    print("💾 STEP 1.7: Saving knowledge base to disk...")
    print("=" * 60)
    
    # Step 1.7a: Create a dictionary containing EVERYTHING we need
    #            This is like packing a suitcase — one file has it all
    knowledge_base = {
        'vectorizer': vectorizer,        # The TF-IDF vectorizer (to vectorize new queries)
        'tfidf_matrix': tfidf_matrix,    # The precomputed TF-IDF matrix (all chunk vectors)
        'chunks': all_chunks,            # The original text chunks (to display in results)
        'metadata': all_metadata,        # Metadata for each chunk (disease, category, severity)
        'categories': sorted(set(m['category'] for m in all_metadata)),   # All unique categories
        'severities': sorted(set(m['severity'] for m in all_metadata)),   # All unique severity levels
        'total_diseases': len(set(m['disease'] for m in all_metadata)),   # Count of unique diseases
        'build_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')             # When the KB was built
    }
    
    # Step 1.7b: Save the dictionary to a pickle file
    #            'wb' means "write binary" — pickle files are binary, not text
    with open(KNOWLEDGE_BASE_OUTPUT, 'wb') as f:
        pickle.dump(knowledge_base, f)
    
    # Step 1.7c: Check the file size (in MB) for reference
    file_size_mb = os.path.getsize(KNOWLEDGE_BASE_OUTPUT) / (1024 * 1024)
    
    print(f"   ✅ Knowledge base saved to: {KNOWLEDGE_BASE_OUTPUT}")
    print(f"   📊 File size: {file_size_mb:.2f} MB")
    print(f"   📊 Total chunks indexed: {len(all_chunks)}")
    print(f"   📊 Total unique diseases: {knowledge_base['total_diseases']}")
    print(f"   📊 Categories: {knowledge_base['categories']}")
    print(f"   📊 Severity levels: {knowledge_base['severities']}")
    print(f"   📊 Built at: {knowledge_base['build_timestamp']}")


# ============================================================
# STEP 1.8: MAIN EXECUTION — Run Everything in Order
# ============================================================

def main():
    """
    STEP 1.8: Main function that orchestrates the entire knowledge base build.
    
    This function calls all the above functions in the correct order:
    1. Load datasets
    2. Clean data
    3. Create enriched chunks
    4. Build TF-IDF index
    5. Save to disk
    
    Think of this as the "conductor" of an orchestra — it tells
    each section when to play.
    """
    
    print("\n" + "🏥" * 30)
    print("   MedAgentixAI — Knowledge Base Builder")
    print("   Building your medical knowledge base...")
    print("🏥" * 30 + "\n")
    
    # Step 1.8a: Record total start time
    total_start = time.time()
    
    # Step 1.8b: STEP 1 — Load the CSV datasets
    medical_df, chunks_df, medical_1000 = load_datasets()
    
    # Step 1.8c: STEP 2 — Clean and preprocess the data
    medical_df, chunks_df, medical_1000 = clean_data(medical_df, chunks_df, medical_1000)
    
    # Step 1.8d: STEP 3 — Create enriched text chunks
    all_chunks, all_metadata = create_enriched_chunks(medical_df, chunks_df, medical_1000)
    
    # Step 1.8e: STEP 4 — Build TF-IDF vector index
    vectorizer, tfidf_matrix = build_tfidf_index(all_chunks)
    
    # Step 1.8f: STEP 5 — Save the knowledge base to disk
    save_knowledge_base(vectorizer, tfidf_matrix, all_chunks, all_metadata, medical_df)
    
    # Step 1.8g: Print total time taken
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print(f"🎉 KNOWLEDGE BASE BUILT SUCCESSFULLY in {total_elapsed:.2f} seconds!")
    print(f"   You can now run the RAG system: streamlit run app.py")
    print("=" * 60)


# ============================================================
# STEP 1.9: SCRIPT ENTRY POINT
# ============================================================
# This block runs ONLY when you execute this file directly:
#   python knowledge_base.py
# It does NOT run when this file is imported by another script.
# ============================================================

if __name__ == "__main__":
    main()
