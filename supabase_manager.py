import os
from supabase import create_client, Client

class SupabaseStorageManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
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
            self.supabase.storage.from_(bucket_name).upload(
                path=remote_path,
                file=f,
                file_options={"upsert": "true"}
            )
        print(f"✅ Successfully uploaded {local_path} to {bucket_name}/{remote_path}")

    def download_vectorstore(self, bucket_name: str, remote_folder: str, local_dir: str):
        """Downloads the vectorstore folder from Supabase to a local directory."""
        import os
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            # List files in the remote folder
            files = self.supabase.storage.from_(bucket_name).list(remote_folder)
            
            if not files:
                print(f"⚠️ No files found in {bucket_name}/{remote_folder}")
                return
            
            print(f"📥 Found {len(files)} files in {bucket_name}/{remote_folder}")
            
            # Download each file
            for file in files:
                file_name = file['name']
                if file_name == ".emptyFolderPlaceholder": 
                    continue
                
                remote_path = f"{remote_folder}/{file_name}"
                local_path = os.path.join(local_dir, file_name)
                
                print(f"📥 Downloading {file_name}...")
                
                try:
                    # Download file
                    res = self.supabase.storage.from_(bucket_name).download(remote_path)
                    
                    # Write to local file
                    with open(local_path, "wb") as f:
                        f.write(res)
                    
                    print(f"✅ Downloaded {file_name} ({len(res)} bytes)")
                    
                except Exception as e:
                    print(f"❌ Failed to download {file_name}: {str(e)}")
                    
        except Exception as e:
            print(f"❌ Error listing files from Supabase: {str(e)}")
            raise

    def download_file(self, bucket_name: str, remote_path: str, local_path: str):
        """Downloads a single file from Supabase Storage."""
        try:
            # Download file
            res = self.supabase.storage.from_(bucket_name).download(remote_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Write to local file
            with open(local_path, "wb") as f:
                f.write(res)
            
            print(f"✅ Downloaded {remote_path} to {local_path} ({len(res)} bytes)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {remote_path}: {str(e)}")
            return False
