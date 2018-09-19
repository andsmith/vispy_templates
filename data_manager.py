import sqlite3

class DataManager(object):

    def __init__(self, db_file):
        self._db_file = db_file
        self._con = sqlite3.connect(self._db_file)
        self._cur = self._con.cursor()
 
    def execute(self, query):
        response = self._cur.execute(query)
        return response.fetchall()

    def execute_and_commit(self, query):
        response = self._con.execute(query)
        self._con.commit()
        return response.fetchall()

def import_baseline_records(table, rec_location, fields):
    pass

if __name__ == "__main__":

    dm = DataManager("test_db.db")

    query = """Create table test_table2 (
                         id INTEGER PRIMARY KEY,
                         name STRING NOT NULL,
                         size float);"""

    try:
        print dm.execute_and_commit(query)
        print "Query succeeded."
    except sqlite3.OperationalError as e:
        print "Query failed:", e.message
