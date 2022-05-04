from dataclasses import dataclass
import psycopg2 as psy
import numpy as np
import pickle
import tensorflow as tf

class RetrieveData:
    def __init__(self,connection,cursor):
        self.conn = connection
        self.cur = cursor

    def retrive_bot_names(self):
        self.cur.execute("SELECT bot_name FROM comments")
        bot_list = []
        for row in self.cur:
            if row[0] not in bot_list:
                bot_list.append(row[0])
        return bot_list
    
    def retrieve_bot_score(self,bot):
        self.cur.execute("SELECT bot_score FROM comments WHERE bot_name = %s",(bot,))
        return self.cur.fetchone()
    
    def retrieve_prepro_train(self):
        self.cur.execute("SELECT * FROM prepro_data")
        rows = self.cur.fetchall()
        data = []
        for row in rows:
            data.append(list(row))
        for row in data:
            arr = pickle.loads(row[2])
            row[2] = tf.convert_to_tensor(arr)
            row[0] = row[0].strip()
            row[1] = float(row[1])
            if row[1] < .5:
                row.append(0)
            if row[1] >= .5:
                row.append(1)
        return data

    def retrieve_prepro_test(self):
        self.cur.execute("SELECT * FROM prepro_data WHERE com_score < .49")
        rows = self.cur.fetchall()
        data = []
        for row in rows:
            data.append(list(row))
        for row in data:
            row[2] = pickle.loads(row[2])
            row[0] = row[0].strip()
        return data


