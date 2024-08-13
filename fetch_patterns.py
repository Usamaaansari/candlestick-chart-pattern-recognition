#### FETCHING PATTERNS FROM DATABASE TABLE #################

import mysql.connector
from config import DATABASE_CONFIG

def get_pattern_counts(table_name):
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = connection.cursor(dictionary=True)

        query = f"""
        SELECT pattern, count
        FROM {table_name}
        """
        cursor.execute(query)

        pattern_counter = {row['pattern']: row['count'] for row in cursor.fetchall()}

        cursor.close()
        connection.close()

        return pattern_counter
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return {}

def get_stocks_by_pattern(table_name, pattern):
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        cursor = connection.cursor(dictionary=True)

        pattern_table = table_name.split('_')[0]+'_'+'patterns' # Name of the table where patterns are stored
        query = f"""
        SELECT c.company_name, c.symbol
        FROM {table_name} c
        JOIN {pattern_table} p ON c.pattern_id = p.id
        WHERE p.pattern = %s
        """
        cursor.execute(query, (pattern,))

        stocks = cursor.fetchall()

        cursor.close()
        connection.close()

        return stocks
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

