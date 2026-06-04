"""
Chatbot API Routes — MedAgentix RAG Chatbot Endpoint

Exposes a POST /api/v1/chatbot endpoint that accepts a user message,
runs it through the MedRAG pipeline (retrieve + generate), and returns
the AI-generated response along with source citations.
"""

import os
import sys
from flask import Blueprint, jsonify, request

chatbot_bp = Blueprint('chatbot', __name__, url_prefix='/api/v1/chatbot')

# ---------------------------------------------------------------------------
# Lazy-loaded singleton for the RAG system
# We initialize it on first request to avoid slowing down Flask startup
# ---------------------------------------------------------------------------
_rag_instance = None


def _get_rag():
    """Lazily initialize and cache the MedRAG instance."""
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    try:
        # Ensure the rag/ package is importable
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from rag.rag_system import MedRAG

        # The knowledge base pickle lives inside the rag/ directory
        kb_path = os.path.join(project_root, 'rag', 'knowledge_base.pkl')
        _rag_instance = MedRAG(knowledge_base_path=kb_path)
        return _rag_instance

    except FileNotFoundError as e:
        print(f"[Chatbot] Knowledge base not found: {e}")
        return None
    except Exception as e:
        print(f"[Chatbot] Failed to initialize RAG system: {e}")
        return None


# ---------------------------------------------------------------------------
# POST  /api/v1/chatbot
# Body: { "message": "user question here" }
# ---------------------------------------------------------------------------
@chatbot_bp.route('', methods=['POST'])
def chatbot_query():
    """
    Accept a user message, run the RAG pipeline, and return the answer.
    This endpoint does NOT require authentication so the chatbot is
    accessible from the landing page and all dashboard views.
    """
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()

    if not user_message:
        return jsonify({
            "success": False,
            "error": "Message is required."
        }), 400

    # Obtain the RAG system (lazy init on first call)
    rag = _get_rag()

    if rag is None:
        return jsonify({
            "success": False,
            "error": "RAG system is not available. Please ensure knowledge_base.pkl exists."
        }), 503

    try:
        # Run the full RAG pipeline: Retrieve → Augment → Generate
        answer, retrieved_chunks = rag.query(user_message, top_k=5)

        # Build a simplified sources list for the frontend
        sources = []
        for chunk in retrieved_chunks:
            meta = chunk.get('metadata', {})
            sources.append({
                "disease": meta.get('disease', 'Unknown'),
                "category": meta.get('category', ''),
                "severity": meta.get('severity', ''),
                "relevance": chunk.get('similarity_score', 0.0)
            })

        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        }), 200

    except Exception as e:
        print(f"[Chatbot] Error processing query: {e}")
        return jsonify({
            "success": False,
            "error": "An internal error occurred while processing your question."
        }), 500


# ---------------------------------------------------------------------------
# GET  /api/v1/chatbot/status
# Quick health-check for the chatbot subsystem
# ---------------------------------------------------------------------------
@chatbot_bp.route('/status', methods=['GET'])
def chatbot_status():
    """Return the initialization status of the RAG chatbot."""
    rag = _get_rag()
    if rag is None:
        return jsonify({
            "status": "unavailable",
            "message": "RAG knowledge base is not loaded."
        }), 503

    stats = rag.get_stats()
    return jsonify({
        "status": "ready",
        "total_chunks": stats.get('total_chunks', 0),
        "total_diseases": stats.get('total_diseases', 0),
        "has_gemini": stats.get('has_gemini', False)
    }), 200
