"""

"""
import sqlite3


class DatabaseManager:

    def __init__(self, db_path):
        """

        Args:
            db_path:
        """
        self.db_path = db_path

    def get_connection(self):

        return sqlite3.connect(self.db_path)

    def execute_query(self, query, params=None):
        """

        Args:
            query: SQL
            params:

        Returns:
            list:
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        results = cursor.fetchall()
        conn.close()

        return results

    def execute_update(self, query, params=None):
        """
        gL

        Args:
            query: SQL
            params:

        Returns:
            int:
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        rowcount = cursor.rowcount
        conn.commit()
        conn.close()

        return rowcount
