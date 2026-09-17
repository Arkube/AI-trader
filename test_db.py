import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', database='postgres')
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = 'trading'")
exists = cur.fetchone()
print(f'trading database exists: {exists is not None}')
if exists:
    conn2 = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', database='trading')
    cur2 = conn2.cursor()
    cur2.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
    ts = cur2.fetchone()
    print(f'TimescaleDB extension: {ts is not None}')
    conn2.close()
conn.close()