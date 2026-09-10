"""
Migration: Replace age column with date_of_birth in the users table.
Uses SQLAlchemy engine directly.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Sayali123@localhost:5432/medagentix_db")

def migrate():
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Check existing columns
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'users' AND column_name IN ('age', 'date_of_birth')"
        ))
        existing = [row[0] for row in result.fetchall()]

        # Add date_of_birth if not exists
        if 'date_of_birth' not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN date_of_birth DATE"))
            print(" [+] Added 'date_of_birth' column to users table.")
        else:
            print(" [-] 'date_of_birth' column already exists, skipping.")

        # Drop old age column if it exists
        if 'age' in existing:
            conn.execute(text("ALTER TABLE users DROP COLUMN age"))
            print(" [+] Dropped old 'age' column from users table.")
        else:
            print(" [-] 'age' column already gone, skipping.")

        conn.commit()
        print("\n Migration complete!")

    engine.dispose()


if __name__ == '__main__':
    migrate()
