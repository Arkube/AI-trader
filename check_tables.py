import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', database='trading')
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
tables = cur.fetchall()
for t in tables:
    print(t[0])
conn.close()