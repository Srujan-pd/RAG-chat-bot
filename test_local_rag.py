import os
from dotenv import load_dotenv

load_dotenv()

# Import after load_dotenv so env vars are set
import rag_engine

print("Testing RAG Engine...\n")

# Load vector store
rag_engine.load_vectorstore_from_supabase()

# Check if loaded (use the correct module-level variable)
if rag_engine.db is None:
    print("❌ Vector store failed to load!")
    print(f"Error: {rag_engine.loading_error}")
    exit(1)

print("\n✅ Vector store loaded! Testing queries...\n")
print("="*70 + "\n")

# Test queries
queries = [
    "What services does Primis Digital offer?",
    "Tell me about AI development",
    "What is DevOps?",
    "How can I contact Primis Digital?"
]

for query in queries:
    print(f"❓ Query: {query}")
    answer = rag_engine.rag_answer(query)
    print(f"💬 Answer: {answer}\n")
    print("-" * 70 + "\n")

print("\n🎉 All tests completed successfully!")
