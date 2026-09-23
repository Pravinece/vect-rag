from config import DB_URL
import psycopg2

EMBEDDING_DIM = 1024

def _init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS source_registry (
                id          SERIAL PRIMARY KEY,
                filename    TEXT UNIQUE NOT NULL,
                table_name  TEXT NOT NULL,
                description TEXT,
                chunk_count INTEGER DEFAULT 0,
                created_at  TIMESTAMP DEFAULT NOW(),
                updated_at  TIMESTAMP DEFAULT NOW()
            )
        """)
    conn.commit()
    print("Tables and indexes ready")

def create_source_table(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id        SERIAL PRIMARY KEY,
                chunk_id  INTEGER NOT NULL,
                content   TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                UNIQUE (chunk_id)
            )
        """)
    conn.commit()

def build_vector_index(conn, table_name: str):
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS idx_{table_name}_embedding")
        cur.execute(f"CREATE INDEX idx_{table_name}_embedding ON {table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)")
    conn.commit()
    print(f"Vector index built for {table_name}")

def get_all_registry(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename, table_name, description, chunk_count, created_at, updated_at FROM source_registry WHERE filename IS NOT NULL")
        rows = cur.fetchall()
    if not rows:
        return []
    return [{"id": r[0], "filename": r[1], "table_name": r[2], "description": r[3], "chunk_count": r[4], "created_at": r[5], "updated_at": r[6]} for r in rows]

def get_registry(conn, filename: str):
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename, table_name, description, chunk_count, created_at, updated_at FROM source_registry WHERE filename = %s", (filename,))
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "filename": row[1], "table_name": row[2], "description": row[3], "chunk_count": row[4], "created_at": row[5], "updated_at": row[6]}

def upsert_registry(conn, filename: str, table_name: str, description: str, chunk_count: int, is_update: bool):
    with conn.cursor() as cur:
        if is_update:
            cur.execute("""
                UPDATE source_registry
                SET description = %s, chunk_count = %s, updated_at = NOW()
                WHERE filename = %s
            """, (description, chunk_count, filename))
        else:
            cur.execute("""
                INSERT INTO source_registry (filename, table_name, description, chunk_count)
                VALUES (%s, %s, %s, %s)
            """, (filename, table_name, description, chunk_count))
    conn.commit()

class Postgres:

    def __init__(self):
        self.conn = None

    def connect(self):
        self.conn = psycopg2.connect(DB_URL)
        print("Connected to Postgres")
        _init_db(self.conn)

    def disconnect(self):
        if self.conn:
            self.conn.close()
            print("Disconnected from Postgres")

postgres = Postgres()