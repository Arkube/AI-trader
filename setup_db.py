import psycopg2

# Create trading database
conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', database='postgres')
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE DATABASE trading")
print("Created trading database")
conn.close()

# Enable TimescaleDB extension
conn2 = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', database='trading')
conn2.autocommit = True
cur2 = conn2.cursor()
cur2.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
print("Enabled TimescaleDB extension")
conn2.close()

# Verify
conn3 = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', database='trading')
cur3 = conn3.cursor()
cur3.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
ts = cur3.fetchone()
print(f'TimescaleDB extension: {ts is not None}')
conn3.close()