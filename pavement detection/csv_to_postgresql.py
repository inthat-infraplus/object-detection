import psycopg2
import csv
import os
import glob

# Database connection parameters
DB_HOST = "localhost"
DB_NAME = "test_pavement"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "1234"

# Function to connect to PostgreSQL database
def connect_to_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

# Main function to ingest data from all CSV files in the specified folder
def ingest_data_from_folder(csv_folder):
    # Connect to the database
    conn = connect_to_db()
    cur = conn.cursor()

    # Construct the full path to the CSV folder
    csv_path = os.path.join(os.getcwd(), csv_folder)

    # Use glob to find all CSV files in the specified folder
    all_csv_files = glob.glob(os.path.join(csv_path, '*.csv'))

    if not all_csv_files:
        print(f"No CSV files found in the folder: {csv_path}")
        cur.close()
        conn.close()
        return

    # Iterate through each CSV file found
    for csv_file_path in all_csv_files:
        print(f"Processing file: {csv_file_path}")
        with open(csv_file_path, 'r') as file:
            data_reader = csv.reader(file)
            next(data_reader)  # Skip the header row

            # Insert each row into the table
            for row in data_reader:
                try:
                    cur.execute("INSERT INTO pavement_ai (name,object_id,pavement_class,pavement_value,x1,y1,x2,y2,confidence) VALUES ( %s, %s, %s, %s,%s, %s, %s, %s, %s)", row)
                except psycopg2.Error as e:
                    print(f"Error inserting row {row} from {csv_file_path}: {e}")
                    conn.rollback() # Rollback the transaction if an error occurs

    # Commit and close connection
    conn.commit()
    cur.close()
    conn.close()
    print("Data ingestion from all CSV files complete.")

if __name__ == "__main__":
    csv_folder_input = input("Please enter the path to the CSV folder: ")
    ingest_data_from_folder(csv_folder_input)