import os
import sys
from dotenv import load_dotenv
from supabase_manager import SupabaseStorageManager

load_dotenv()

def verify_files_exist():
    """Verify local vector store files exist"""
    required_files = [
        "vectorstore/index.faiss",
        "vectorstore/index.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run create_vectorstore.py or rag_ingest.py first!")
        return False
    
    # Show file sizes
    print("📊 Local files found:")
    for file_path in required_files:
        size = os.path.getsize(file_path)
        print(f"   - {file_path}: {size:,} bytes")
    
    return True

def sync():
    print("=" * 50)
    print("📤 Supabase Vector Store Upload Tool")
    print("=" * 50)
    
    # Step 1: Verify local files
    if not verify_files_exist():
        sys.exit(1)
    
    # Step 2: Check environment variables
    print("\n🔍 Checking environment...")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    bucket_name = os.getenv("SUPABASE_BUCKET_NAME", "vectorstore-bucket")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials in .env file")
        print("   Required: SUPABASE_URL, SUPABASE_KEY")
        sys.exit(1)
    
    print(f"✅ Supabase URL: {supabase_url[:20]}...")
    print(f"✅ Bucket: {bucket_name}")
    
    # Step 3: Upload
    print("\n📤 Starting sync to Supabase...")
    storage = SupabaseStorageManager()
    
    success_count = 0
    fail_count = 0
    
    # Upload main files
    files_to_upload = [
        ("vectorstore/index.faiss", "vectorstore/index.faiss"),
        ("vectorstore/index.pkl", "vectorstore/index.pkl")
    ]
    
    for local_path, remote_path in files_to_upload:
        if storage.upload_file(local_path, remote_path, bucket_name):
            success_count += 1
        else:
            fail_count += 1
    
    # Step 4: Verify upload
    print("\n🔍 Verifying upload...")
    files = storage.list_files(bucket_name, "vectorstore")
    
    if len(files) >= 2:
        print(f"✅ Found {len(files)} files in vectorstore folder")
        
        # Check for specific files
        remote_names = [f.get('name', '') for f in files]
        if 'index.faiss' in str(remote_names) and 'index.pkl' in str(remote_names):
            print("✅ Both vector store files verified in Supabase")
        else:
            print("⚠️ Files may be incomplete in Supabase")
    else:
        print(f"⚠️ Only found {len(files)} files in Supabase")
    
    print("\n" + "=" * 50)
    if success_count == 2 and fail_count == 0:
        print("🎉 All files synced successfully to Supabase!")
    else:
        print(f"⚠️ Upload complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 50)

if __name__ == "__main__":
    sync()
