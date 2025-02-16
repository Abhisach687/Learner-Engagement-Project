import sqlite3

# Connect to the database
conn = sqlite3.connect('tuning.db')
cursor = conn.cursor()

# Execute a query (e.g., list trials)
cursor.execute("SELECT * FROM trials;")
rows = cursor.fetchall()

# Print the results
for row in rows:
    print(row)

# Close the connection
conn.close()