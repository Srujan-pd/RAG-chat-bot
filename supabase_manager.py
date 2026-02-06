import os
import logging
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseStorageManager:
    def __init__(self):
        """Initialize Supabase client"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        logger.info(f"🔗 Initializing Supabase client for URL: {supabase_url[:30]}...")
        
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to create Supabase client: {e}")
            raise
    
    def download_file(self, remote_path: str, local_path: str, bucket_name: str) -> bool:
        """Download file from Supabase Storage"""
        try:
            logger.info(f"⬇️  Downloading {remote_path} from bucket {bucket_name}...")
            
            # Download file content
            response = self.client.storage.from_(bucket_name).download(remote_path)
            
            if response is None:
                logger.error(f"❌ No response received for {remote_path}")
                return False
            
            # Write to local file
            with open(local_path, 'wb') as f:
                f.write(response)
            
            size = os.path.getsize(local_path)
            logger.info(f"✅ Downloaded {remote_path}: {size:,} bytes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed for {remote_path}: {str(e)}")
            return False
    
    def upload_file(self, local_path: str, remote_path: str, bucket_name: str) -> bool:
        """Upload file to Supabase Storage"""
        try:
            logger.info(f"⬆️  Uploading {local_path} to {remote_path}...")
            
            with open(local_path, 'rb') as f:
                file_content = f.read()
            
            # Upload with upsert to overwrite if exists
            self.client.storage.from_(bucket_name).upload(
                path=remote_path,
                file=file_content,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            
            logger.info(f"✅ Uploaded {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {local_path}: {str(e)}")
            return False
    
    def list_files(self, bucket_name: str, folder: str = "") -> list:
        """List files in a bucket folder"""
        try:
            files = self.client.storage.from_(bucket_name).list(folder)
            return files
        except Exception as e:
            logger.error(f"❌ List files failed: {str(e)}")
            return []
