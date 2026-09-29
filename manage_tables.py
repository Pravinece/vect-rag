import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(os.getenv("DB_URL"))
cur = conn.cursor()

# ── LIST tables ──────────────────────────────────────────────
cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
print("Tables:", cur.fetchall())

# ── TRUNCATE (clears data, keeps structure) ───────────────────
# cur.execute("TRUNCATE TABLE documents CASCADE")
# cur.execute("TRUNCATE TABLE source_registry CASCADE")
# cur.execute("TRUNCATE TABLE docs_pravind_profile_and_skills CASCADE")

# ── TRUNCATE ALL tables at once ───────────────────────────────
# cur.execute("TRUNCATE TABLE documents, source_registry, docs_pravind_profile_and_skills CASCADE")

# ── DROP (deletes table completely) ───────────────────────────
# cur.execute("DROP TABLE IF EXISTS documents CASCADE")
# cur.execute("DROP TABLE IF EXISTS source_registry CASCADE")
# cur.execute("DROP TABLE IF EXISTS docs_pravind_profile_and_skills CASCADE")

# ── DROP ALL tables at once ───────────────────────────────────
# cur.execute("DROP TABLE IF EXISTS documents, source_registry, docs_pravind_profile_and_skills CASCADE")

conn.commit()
cur.close()
conn.close()
print("Done.")
