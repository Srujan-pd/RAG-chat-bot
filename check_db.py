from sqlalchemy import create_engine, inspect
import os
from dotenv import load_dotenv

load_dotenv()  # must be before accessing DATABASE_URL

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL)
inspector = inspect(engine)

print("Tables in database:", inspector.get_table_names())

