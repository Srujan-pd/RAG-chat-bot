import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore():
    """Create FAISS vector store from scraped data"""
    
    # 1. Load scraped data
    print("📖 Loading scraped data...")
    with open('data/scraped_data.json', 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    print(f"✅ Loaded {len(pages)} pages")
    
    # 2. Combine all text
    print("\n📝 Combining text from all pages...")
    all_text = []
    for page in pages:
        page_text = f"SOURCE: {page['url']}\nTITLE: {page['title']}\n\n{page['content']}\n\n"
        all_text.append(page_text)
    
    combined_text = "\n---\n".join(all_text)
    print(f"✅ Combined text length: {len(combined_text):,} characters")
    
    # 3. Split into chunks
    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(combined_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # 4. Save chunks to readable file (for verification)
    print("\n💾 Saving chunks to readable file...")
    os.makedirs('data', exist_ok=True)
    with open('data/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Chunks saved to: data/chunks.json (you can read this!)")
    
    # 5. Create embeddings
    print("\n🔧 Creating embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embeddings model ready")
    
    # 6. Create FAISS vector store
    print("\n🧠 Creating FAISS vector store...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # 7. Save vector store
    print("\n💾 Saving vector store...")
    os.makedirs('vectorstore', exist_ok=True)
    vectorstore.save_local('vectorstore')
    
    # Check file sizes
    faiss_size = os.path.getsize('vectorstore/index.faiss')
    pkl_size = os.path.getsize('vectorstore/index.pkl')
    
    print(f"\n✅ Vector store created successfully!")
    print(f"   📁 vectorstore/index.faiss: {faiss_size:,} bytes")
    print(f"   📁 vectorstore/index.pkl: {pkl_size:,} bytes")
    print(f"\n📊 Summary:")
    print(f"   - Pages scraped: {len(pages)}")
    print(f"   - Text chunks: {len(chunks)}")
    print(f"   - Vector dimensions: 384 (MiniLM)")
    
    return vectorstore

if __name__ == "__main__":
    create_vectorstore()
