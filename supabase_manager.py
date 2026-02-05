import os
from supabase import create_client, Client

class SupabaseStorageManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        # Ensure you are using the SERVICE_ROLE_KEY for uploads
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY not set in environment")
        self.supabase: Client = create_client(url, key)

    def upload_file(self, local_path: str, remote_path: str, bucket_name: str = "vectorstore-bucket"):
        """Uploads a local file to Supabase Storage."""
        if not os.path.exists(local_path):
            print(f"⚠️ Warning: Local file {local_path} not found.")
            return

        with open(local_path, "rb") as f:
            # We use upsert=True to overwrite existing files
            self.supabase.storage.from_(bucket_name).upload(
                path=remote_path,
                file=f,
                file_options={"upsert": "true"}
            )
        print(f"✅ Successfully uploaded {local_path} to {bucket_name}/{remote_path}")

    def download_vectorstore(self, bucket_name: str, remote_folder: str, local_dir: str):
        """Downloads the vectorstore folder from Supabase to a local directory."""
        os.makedirs(local_dir, exist_ok=True)
        files = self.supabase.storage.from_(bucket_name).list(remote_folder)
        
        if not files:
            print(f"⚠️ No files found in {bucket_name}/{remote_folder}")
            return

        for file in files:
            file_name = file['name']
            if file_name == ".emptyFolderPlaceholder": continue
            
            remote_path = f"{remote_folder}/{file_name}"
            local_path = os.path.join(local_dir, file_name)
            
            print(f"📥 Downloading {file_name}...")
            with open(local_path, "wb+") as f:
                res = self.supabase.storage.from_(bucket_name).download(remote_path)
                f.write(res)
        print(f"✅ Vectorstore synced to {local_dir}")
