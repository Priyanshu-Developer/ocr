import psycopg2

# PostgreSQL connection details
DB_HOST = "dpg-d1vnkrndiees73brp680-a.oregon-postgres.render.com"
DB_PORT = "5432"
DB_NAME = "client_jo5r"
DB_USER = "priyanshu"
DB_PASSWORD = "fw0lwMwJpbDYuTW9rwlBHB8w2HLAVoK8"  # Change to your DB password

try:
    # Attempt connection
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    # If successful
    print("✅ PostgreSQL connection successful!")
    print("Connected to database:", conn.get_dsn_parameters()['dbname'])
    conn.close()

except psycopg2.Error as e:
    # If connection fails
    print("❌ PostgreSQL connection failed!")
    print("Error:", e)


# postgresql://priyanshu:fw0lwMwJpbDYuTW9rwlBHB8w2HLAVoK8@dpg-d1vnkrndiees73brp680-a.oregon-postgres.render.com/client_jo5r