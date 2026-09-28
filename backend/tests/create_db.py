import asyncio
import asyncpg
import urllib.parse

async def run():
    try:
        pw = urllib.parse.quote_plus('#123Wissen')
        conn = await asyncpg.connect(f'postgresql://postgres:{pw}@135.181.19.83:5432/postgres')
        
        # Check if database exists
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = 'voice_agent'")
        if not exists:
            await conn.execute("CREATE DATABASE voice_agent")
            print("Database 'voice_agent' created.")
        else:
            print("Database 'voice_agent' already exists.")
            
        await conn.close()
    except Exception as e:
        print("Error:", e)

asyncio.run(run())
