import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

def extract_contact_info(text):
    """Extract contact information from text"""
    contacts = []
    
    # Phone patterns
    phone_patterns = [
        r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
        r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\s*(?:ext|x|extension)?\s*\d*',
    ]
    
    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            contacts.extend([f"📞 Phone: {phone}" for phone in phones])
    
    emails = re.findall(email_pattern, text)
    if emails:
        contacts.extend([f"✉️ Email: {email}" for email in emails])
    
    return contacts

def create_vectorstore():
    """Create FAISS vector store from scraped data"""
    
    # 1. Load scraped data
    print("📖 Loading scraped data...")
    
    # Check if scraped_data.json exists
    if not os.path.exists('data/scraped_data.json'):
        print("❌ No scraped data found! Run scraper.py first.")
        return
    
    with open('data/scraped_data.json', 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    print(f"✅ Loaded {len(pages)} pages")
    
    # Count contact pages
    contact_pages = [p for p in pages if 'contact' in p.get('url', '').lower() or 
                     any(kw in p.get('url', '').lower() for kw in ['about', 'company', 'support'])]
    print(f"📞 Found {len(contact_pages)} potential contact/about pages")
    
    # 2. Process pages with contact info highlighting
    print("\n📝 Processing pages and highlighting contact info...")
    all_text = []
    
    for page in pages:
        # Extract contact info
        contacts = extract_contact_info(page['content'])
        
        # Build page text with contact info highlighted
        page_text = f"SOURCE: {page['url']}\nTITLE: {page['title']}\n"
        
        # Add highlighted contact info if found
        if contacts:
            page_text += "CONTACT INFORMATION:\n"
            for contact in contacts:
                page_text += f"{contact}\n"
            page_text += "\n"
        
        page_text += f"CONTENT:\n{page['content']}\n\n"
        all_text.append(page_text)
        
        if contacts:
            print(f"   ✅ Contact info found on: {page['url']}")
            for contact in contacts:
                print(f"      {contact}")
    
    combined_text = "\n---\n".join(all_text)
    print(f"✅ Combined text length: {len(combined_text):,} characters")
    
    # 3. Split into chunks with better strategy for contact info
    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,  # Increased overlap
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    chunks = splitter.split_text(combined_text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Check how many chunks contain contact info
    contact_chunks = 0
    for chunk in chunks:
        if any(keyword in chunk.lower() for keyword in ['📞', '✉️', 'phone:', 'email:', '@', 'contact']):
            contact_chunks += 1
    
    print(f"📞 Chunks with contact information: {contact_chunks}/{len(chunks)} ({contact_chunks/len(chunks)*100:.1f}%)")
    
    # 4. Save chunks to readable file
    print("\n💾 Saving chunks to readable file...")
    os.makedirs('data', exist_ok=True)
    
    # Save with metadata
    chunks_data = {
        "total_chunks": len(chunks),
        "contact_chunks": contact_chunks,
        "chunks": chunks[:20],  # First 20 chunks for preview
        "all_chunks": chunks  # All chunks
    }
    
    with open('data/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Chunks saved to: data/chunks.json")
    
    # 5. Create embeddings
    print("\n🔧 Creating embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
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
    if os.path.exists('vectorstore/index.faiss'):
        faiss_size = os.path.getsize('vectorstore/index.faiss')
        pkl_size = os.path.getsize('vectorstore/index.pkl')
        
        print(f"\n✅ Vector store created successfully!")
        print(f"   📁 vectorstore/index.faiss: {faiss_size:,} bytes")
        print(f"   📁 vectorstore/index.pkl: {pkl_size:,} bytes")
    
    print(f"\n📊 Summary:")
    print(f"   - Pages scraped: {len(pages)}")
    print(f"   - Contact pages: {len(contact_pages)}")
    print(f"   - Text chunks: {len(chunks)}")
    print(f"   - Contact chunks: {contact_chunks}")
    print(f"   - Vector dimensions: 384 (MiniLM)")
    
    # Test search for contact info
    print("\n🧪 Testing contact info search...")
    test_queries = ["contact phone number", "email support", "call primis"]
    
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\n   Query: '{query}'")
        print(f"   Found: {len(results)} results")
        for i, doc in enumerate(results):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"     {i+1}. {preview}...")
    
    return vectorstore

if __name__ == "__main__":
    create_vectorstore()
