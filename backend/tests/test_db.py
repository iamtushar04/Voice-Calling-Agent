import asyncio
import asyncpg
import urllib.parse

async def test_conn():
    try:
        # The # in password must be URL encoded as %23
        password = urllib.parse.quote_plus("#123Wissen")
        conn = await asyncpg.connect(f'postgresql://postgres:{password}@135.181.19.83:5432/postgres')
        val = await conn.fetchval('SELECT version()')
        print('SUCCESS:', val)
        await conn.close()
    except Exception as e:
        print('ERROR:', e)

asyncio.run(test_conn())
