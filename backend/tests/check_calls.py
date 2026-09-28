import asyncio
import asyncpg
import urllib.parse

async def test():
    try:
        pw = urllib.parse.quote_plus('#123Wissen')
        conn = await asyncpg.connect(f'postgresql://postgres:{pw}@135.181.19.83:5432/voice_agent')
        rows = await conn.fetch('SELECT * FROM calls ORDER BY created_at DESC LIMIT 1')
        print(dict(rows[0]) if rows else 'No calls')
        await conn.close()
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
