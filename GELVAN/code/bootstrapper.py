from random import randint as rand
from tomli import load
from shutil import rmtree
from os import mkdir
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa,padding
from cryptography.hazmat.primitives import serialization,hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key,load_pem_public_key
from cryptography.fernet import Fernet
from pickle import dump as pdump
from shutil import copyfile
from json import load as jload
from web3 import Web3
from os import listdir
from os import remove as rm
from random import randint as rand
from datetime import datetime as dt
import ipfs_api

import pandas as pd
import toml

class Bootstrap:
    def __init__(self):
        print("Hola Amiago")
        with open("config.toml","rb") as f:
            self.CONFIG = load(f)
        self.CONFIG = self.CONFIG["ENCODER"]
        
    def data_preparation(self):

        PATH =  self.CONFIG["TRAIN_DATA_PATH"]
        NUM_PARITIONS = self.CONFIG["NUM_DATAPROVIDER"]
        ENCRYPTION = self.CONFIG["ENCRYPTION"]
        UNIFORM = self.CONFIG["UNIFORM"]
        VARIENCE = self.CONFIG["VARIENCE"]
        BASE = self.CONFIG["BASE_PATH"]
        data = pd.read_csv(PATH)
        data=data.drop_duplicates(keep=False)
        data=data.sample(frac=1)
        data_len = data.shape[0]
        mini = data_len//(NUM_PARITIONS)
        maxi = mini+VARIENCE
        PARTITIONS=[]
        while (data.shape[0]>maxi) :
            
            if not UNIFORM and data.shape[0]>maxi:
                maxi=mini+rand(10,VARIENCE)
                part = data.sample(n=rand(mini-VARIENCE,maxi))

            elif(data_len//(NUM_PARITIONS) < data.shape[0]):
                maxi=data_len//(NUM_PARITIONS)
                part = data.sample(n=data_len//(NUM_PARITIONS))
            temp = pd.concat([part,data])
            data=temp.drop_duplicates(keep=False)
            PARTITIONS.append(part)

        if not UNIFORM:

            if(data.shape[0]>mini):
                PARTITIONS.append(data[:mini])
            while len(PARTITIONS)>NUM_PARITIONS:
                PARTITIONS.pop()
        else:
            while len(PARTITIONS) != NUM_PARITIONS:
                PARTITIONS.append(data[:mini])
                data=data[:mini]
        print("\tGenerated Partitions",len(PARTITIONS),NUM_PARITIONS)

        rmtree(BASE,ignore_errors=True)#CAREFULL chagne the name
        mkdir(BASE)
        
        
        for y,x in enumerate(range(self.CONFIG["NUM_DATAPROVIDER"])):  
           
            PARTITIONS[x].to_csv(BASE+str(y+1)+".csv",index=False) #Creating CSVs            
            #copyfile("partitioned_data/"+str(y)+".csv",self.BASE+"/part_data/"+str(y)+".csv") #Copying Them For Later Refernce
            


        print("[BOOTSTRAPPER]PARTITION READY")

a = Bootstrap()
a.data_preparation()