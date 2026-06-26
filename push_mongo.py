import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from churn_reactivation_structure.exception.exception import ChurnReactivationStructure
from churn_reactivation_structure.logging.logger import logging

class ChurnReactivationDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ChurnReactivationStructure(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise ChurnReactivationStructure(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            batch_size = 1000

            for i in range(0, len(self.records), batch_size):
                batch = self.records[i:i+batch_size]
                self.collection.insert_many(batch)
                print(f"Inserted {i + len(batch)} records")            
            
            return(len(self.records))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(type(e))
            print(e)
            raise ChurnReactivationStructure(e,sys)
        
if __name__=='__main__':
    FILE_PATH="churn_reactivation_data/reactivation_v6_final.csv"
    DATABASE="MuhammadHilmy"
    Collection="ChurnReactivationData"
    networkobj=ChurnReactivationDataExtract()
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    #print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
        


