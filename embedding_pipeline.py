"""Embedding pipeline for RAG.

Splits transcript text into chunks, generates embeddings using
all-MiniLM-L6-v2 via sentence-transformers, and stores in pgvector.
"""

import os
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Try to import sentence-transformers; fall back to a simple approach
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = _model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

except ImportError:
    import hashlib

    def embed(texts: list[str]) -> list[list[float]]:
        """Fallback: deterministic pseudo-embeddings for dev/testing."""
        np.random.seed(42)
        dummy = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            dummy.append(rng.randn(384).tolist())
        return dummy


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def process_video(video_id: str):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Get transcript
    result = supabase.table("transcripts") \
        .select("full_text, segments") \
        .eq("video_id", video_id) \
        .single() \
        .execute()

    if not result.data:
        print(f"No transcript found for video {video_id}")
        return

    full_text = result.data["full_text"]
    if not full_text:
        print(f"Empty transcript for video {video_id}")
        return

    # Chunk and embed
    chunks = chunk_text(full_text)
    embeddings = embed(chunks)

    # Delete old chunks
    supabase.table("transcript_chunks") \
        .delete() \
        .eq("video_id", video_id) \
        .execute()

    # Insert new chunks
    rows = [
        {
            "video_id": video_id,
            "chunk_index": i,
            "content": chunks[i],
            "embedding": embeddings[i],
        }
        for i in range(len(chunks))
    ]

    for row in rows:
        supabase.table("transcript_chunks").insert(row).execute()

    print(f"Stored {len(rows)} chunks for video {video_id}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python embedding_pipeline.py <video_id>")
        sys.exit(1)
    process_video(sys.argv[1])
