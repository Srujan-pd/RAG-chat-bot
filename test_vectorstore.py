import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# Remove conflicting env var

def test_rag():
    """Test the RAG system locally"""
    
    print("🧪 Testing RAG System\n")
    
    # 1. Load embeddings
    print("📦 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embeddings loaded")
    
    # 2. Load vector store
    print("\n🧠 Loading vector store...")
    vectorstore = FAISS.load_local(
        'vectorstore',
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded")
    
    # 3. Test queries
    queries = [
        "What services does Primis Digital offer?",
        "Tell me about AI development",
        "What is DevOps?",
        "How can I contact Primis Digital?"
    ]
    
    print("\n" + "="*70)
    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 70)
        
        # Search
        docs = vectorstore.similarity_search(query, k=3)
        
        print(f"📚 Found {len(docs)} relevant chunks:\n")
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"{i}. {content}\n")
        
        # Get context for AI
        context = "\n\n".join(d.page_content for d in docs)
        
        # Call Gemini
        print("🤖 Asking Gemini...\n")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            client = genai.Client()
            prompt = f"""Answer using ONLY this information:

{context}

Question: {query}
Answer:"""
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            print(f"💬 Answer: {response.text}\n")
        else:
            print("⚠️  GEMINI_API_KEY not set, skipping AI response\n")
        
        print("="*70)

if __name__ == "__main__":
    
    test_rag()
