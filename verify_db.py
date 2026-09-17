from database.db import read_sql, get_engine
from sqlalchemy import text

# Test 1: Check tables exist
with get_engine().connect() as conn:
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
    tables = [row[0] for row in result]
    print('Tables created:', tables)

# Test 2: Try a simple query
try:
    df = read_sql('SELECT 1 as test')
    print('Read SQL works:', df.shape)
except Exception as e:
    print('Read SQL error:', e)

# Test 3: Check specific tables
for table in ['tick_data', 'minute_candles', 'symbol_master', 'trade_log']:
    try:
        df = read_sql(f'SELECT COUNT(*) as cnt FROM {table}')
        print(f'{table}: {df.iloc[0]["cnt"]} rows')
    except Exception as e:
        print(f'{table}: ERROR - {e}')

print('Database verification complete!')