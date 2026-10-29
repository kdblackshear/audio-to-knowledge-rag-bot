import os
import whisper
import chromadb
from chromadb.utils import embedding_functions
import ollama

# ---------------------------------------------------------------------------
# 1. TRANSCRIBE AUDIO WITH WHISPER (LOCAL)
# ---------------------------------------------------------------------------
def transcribe_audio(file_path: str):
    """Transcribes audio file using OpenAI Whisper (runs locally)."""
    print("🎙️ Loading Whisper model (base)...")
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    
    print("⏳ Transcribing audio (this may take a minute)...")
    result = model.transcribe(file_path)
    return result["segments"]


def format_timestamp(seconds: float) -> str:
    """Helper to format seconds into HH:MM:SS or MM:SS format."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

# ---------------------------------------------------------------------------
# 2. VECTOR DB INDEXING (CHROMADB)
# ---------------------------------------------------------------------------
def build_vector_db(segments):
    """Indexes transcript segments into ChromaDB with metadata timestamps."""
    print("⚡ Creating vector database...")
    chroma_client = chromadb.Client()
    
    # Use default embedding function
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.create_collection(
        name="podcast_transcript", 
        embedding_function=embedding_fn
    )

    documents = []
    metadatas = []
    ids = []

    for i, seg in enumerate(segments):
        start_time = format_timestamp(seg["start"])
        end_time = format_timestamp(seg["end"])
        
        # Format text payload including visual timestamp tag
        text_content = f"[{start_time} - {end_time}] {seg['text'].strip()}"
        
        documents.append(text_content)
        metadatas.append({
            "start": start_time,
            "end": end_time,
            "raw_seconds": seg["start"]
        })
        ids.append(f"segment_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Indexed {len(documents)} transcript chunks successfully.")
    return collection

# ---------------------------------------------------------------------------
# 3. RAG SEARCH & LOCAL LLM ANSWER GENERATION (OLLAMA)
# ---------------------------------------------------------------------------
def ask_question(collection, query: str):
    """Retrieves context from ChromaDB and generates a local LLM response using Ollama."""
    # Retrieve top 4 most relevant audio chunks
    results = collection.query(query_texts=[query], n_results=4)
    retrieved_chunks = results["documents"][0]
    
    # Construct context block
    context = "\n".join(retrieved_chunks)
    
    prompt = f"""
You are an intelligent audio/podcast assistant. Answer the user's question accurately using ONLY the transcript excerpts provided below.
Always include the relevant timestamp brackets (e.g., [MM:SS]) when quoting or citing information so the user knows where in the audio to listen.

Transcript Context:
{context}

Question: {query}
Answer:
"""

    # Call local Ollama model
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.2}
    )

    return response['message']['content']

# ---------------------------------------------------------------------------
# 4. MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    audio_file = "lecture.mp4"  # Pass path to your MP4/audio file
    
    if not os.path.exists(audio_file):
        print(f"⚠️ Please provide a valid audio file path (currently trying '{audio_file}').")
    else:
        # Step 1: Transcribe
        segments = transcribe_audio(audio_file)
        
        # Step 2: Index into ChromaDB
        db = build_vector_db(segments)
        
        # Step 3: Interactive Q&A loop
        print("\n🎉 Ready! Ask questions about your audio (type 'exit' to quit).\n")
        while True:
            query = input("\n❓ Ask a question: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            answer = ask_question(db, query)
            print("\n🤖 Answer:")
            print(answer)