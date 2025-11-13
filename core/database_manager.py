"""
pn“¡h - €„pn“Í\¥ã
"""
import sqlite3


class DatabaseManager:
    """pn“¡h"""

    def __init__(self, db_path):
        """
        Ë

        Args:
            db_path: pn“ï„
        """
        self.db_path = db_path

    def get_connection(self):
        """·Öpn“Þ¥"""
        return sqlite3.connect(self.db_path)

    def execute_query(self, query, params=None):
        """
        gLåâ

        Args:
            query: SQLåâíå
            params: Âp

        Returns:
            list: åâÓœ
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
        gLô°

        Args:
            query: SQLô°íå
            params: Âp

        Returns:
            int: ×qÍ„Lp
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
