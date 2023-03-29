import sqlite3

# Connect to the database
conn = sqlite3.connect('example-study.db')

# Create a cursor object
cursor = conn.cursor()

# Execute a SELECT query
cursor.execute("SELECT * FROM trial_params")

# Fetch the results and print them
results = cursor.fetchall()
for row in results:
    print(row)
    print("\n\n\n")

# Close the cursor and connection
cursor.close()
conn.close()
