from database.db import init_db
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

print("Initializing database...")
init_db()
print("Done!")