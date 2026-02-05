import os
from dotenv import load_dotenv
from supabase_manager import SupabaseStorageManager

load_dotenv()

def sync():
    print("📤 Starting sync to Supabase...")
    storage = SupabaseStorageManager()
    bucket = "vectorstore-bucket"
    
    # Path inside bucket: vectorstore/filename
    # Make sure the local folder 'vectorstore' exists and contains your files
    storage.upload_file("vectorstore/index.faiss", "vectorstore/index.faiss", bucket)
    storage.upload_file("vectorstore/index.pkl", "vectorstore/index.pkl", bucket)
    
    print("🎉 All files synced to Supabase!")

if __name__ == "__main__":
    sync()
