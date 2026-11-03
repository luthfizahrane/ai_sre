"""PostgreSQL connection abstraction with pgvector codec registration."""
from academic_kg.core.config import Settings


class ConnectionFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def connect(self):
        import asyncpg
        from pgvector.asyncpg import register_vector
        s = self.settings
        conn = await asyncpg.connect(
            host=s.db_host, port=s.db_port, user=s.db_user,
            password=s.db_password, database=s.db_name,
        )
        try:
            await register_vector(conn)
        except Exception:
            await conn.close()
            raise
        return conn
