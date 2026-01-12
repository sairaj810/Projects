import socketio
import sys 
import ipfs_api
import pandas as pd
from time import sleep
from random import randint
from tomli import load
from json import load as jload
from pickle import loads as pload
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa,padding
from cryptography.hazmat.primitives import serialization,hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from web3 import Web3
from cryptography.fernet import Fernet
from expert_util import Model
from io import StringIO  
from datetime import datetime as dt
import warnings
warnings.filterwarnings("ignore")


sio = socketio.Client()
OPINION = 9685

CONSENSUS = 0
EXPERT = ""
@sio.event
def connect():
    while not sio.connected:
        continue
    print("[Expert]Connected Just Now. Emiting the ready signal")
    sio.emit("ready")

@sio.event
def disconnect():
    print('disconnected from server')
    print(EXPERT.PREDICTIONS)
    sio.shutdown()
    #SHOULD CALL THE SC UPDATING HERE
    

@sio.event
def handle_error(data):
    raise Exception(data) 

@sio.event
def give_opinion(data):
    #print("\t[REQUEST RECIEVED] RECIEVED OPINION FOR OPINION")
    data["opinion"] = CONSENSUS.OPINION
    sio.emit("opinion_handler",data)

@sio.event
def query_opinion(data):
    CONSENSUS.QUERY_OPINION = data
    #print("\t\t\t[GOT QUERY], MOA MOA",QUERY_OPINION)
@sio.event
def end():
    print("Done")
    sio.shutdown()

@sio.event
def write_to_SC():
    op = ""
    for x in EXPERT.PREDICTIONS:
        op+=str(x)
    print("[EXPERT} Writing Stuff to Smart Contract",op)
    #EXPERT.CONTRACT.functions.updatePredictions()
    tra1 = EXPERT.CONTRACT.functions.updatePredictions(EXPERT.NUMBER,op).transact()
    tx_receipt = EXPERT.CONN.eth.wait_for_transaction_receipt(tra1)
    EXPERT.GAS_USED += tx_receipt["gasUsed"]
    print("COUNTER VALUE is:",EXPERT.CONTRACT.functions.counter().call())
    to_write = str(EXPERT.NUMBER)+";"+str(EXPERT.GAS_USED)+";"+str(EXPERT.MODEL_WEIGHT)
    with open(str(EXPERT.NUMBER)+".txt","w") as f:
        f.write(to_write)
    sio.emit("sc_done")
    sio.sleep(5)
    sio.disconnect()
    

@sio.event
def start_consensus(data):
    #This Plays crucial Role.
    print("[STARTING CONSENSUS]Got Permission to Start Consensus")
    #print("This Plays Crucial Role",data)
    if data != 0:
        opinion = EXPERT.MODEL.predict(data["Data"])
        EXPERT.PREDICTIONS.append(opinion)
        CONSENSUS.OPINION = opinion
        CONSENSUS.INITIAL_OPINION = opinion
    #Reset the counter here, repeat counter
    CONSENSUS.REPEAT_COUNTER = 0    
    CONSENSUS.consensus()
    sio.emit("done")

class Consensus:
    def __init__(self,OPINION,REPEAT_LIMIT,config):
        self.QUERY_OPINION = 0 #Stores the Opinion of the Query
        self.OPINION = OPINION
        self.REPEAT_COUNTER = 0
        self.PROTOCOL_DETAILS = config
        self.UNDECIDED = True  #Used in the consensus method, Untill there is no single opinion
        self.REPEAT_LIMIT = REPEAT_LIMIT  
    def get_query_opinion(self):
        
        self.QUERY_OPINION = 9512
        self.REPEAT_COUNTER+=1
        print("\t[NEW ROUND START] ROUND NUMBER:","REPEAT_COUNTER::",self.REPEAT_COUNTER,self.REPEAT_LIMIT) 
        if(self.REPEAT_COUNTER > self.REPEAT_LIMIT):
            print("IT IS THRASHING SO SEND A MESSAGE TO THE MANAGER BRO")
            return          
        sio.emit("get_query",{'query_size':self.PROTOCOL_DETAILS["K"]})
        while self.QUERY_OPINION == 9512:
            continue  
        #print("STUCK HERE,",self.QUERY_OPINION)

    
    def consensus(self):  

        #print("[STARTED CONSENSUS] Protocol Used:",self.PROTOCOL_DETAILS["NAME"])
       
        
        
        if(self.PROTOCOL_DETAILS["NAME"] == "SLUSH"):
            counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])] #Holds the counter of each opinion. REMEMBER, INDEX is the CLASS
            for x in range(self.PROTOCOL_DETAILS["NUM_ROUNDS"]): 
                counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]
                self.get_query_opinion()
                if(self.QUERY_OPINION == 9512): #If the limit is exceeded then it will be triggered
                    self.OPINION = self.INITIAL_OPINION
                    break
                for op in self.QUERY_OPINION:
                    counter[op]+=1
                #print("\t\t[COUNTER] Counter:",counter)
                if(max(counter) > self.PROTOCOL_DETAILS["ALPHA"]):
                    self.OPINION = counter.index(max(counter))
                    #print("\t\t[OPINION CHANGE] NEW OPINION:", OPINION)
            sio.emit("consensus_decision",{"OPINION":self.OPINION})

        
                
        if(self.PROTOCOL_DETAILS["NAME"] == "SNOWBALL"):
            #Just Need to Add the Counter

            op,prev_op = self.OPINION,self.OPINION
            confidence_counter = 0
            counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]# Different Purpose Counter
            self.UNDECIDED = True
            majority = False
            max_c = 0
            while self.UNDECIDED:
                counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]
                majority = False
                self.get_query_opinion()
                if(self.QUERY_OPINION == 9512): #If the limit is exceeded then it will be triggered
                    self.OPINION = self.INITIAL_OPINION
                    sio.emit("consensus_decision",{"OPINION":op})
                    break
                for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"]):
                    if(self.QUERY_OPINION.count(x) > self.PROTOCOL_DETAILS["ALPHA"]):
                        majority = True
                        counter[x]+=1

                if(max(counter) != 0):
                    op = counter.index(max(counter))
                else:
                    continue

                if(op != prev_op):
                    prev_op = op
                    confidence_counter=0
                else:
                    confidence_counter+=1
                
                if(confidence_counter > self.PROTOCOL_DETAILS["BETA"]):
                    self.OPINION = op
                    sio.emit("consensus_decision",{"OPINION":op})
                    self.UNDECIDED = False
                    
                if(majority == False):
                    confidence_counter = 0

        if(self.PROTOCOL_DETAILS["NAME"] == "SNOWFLAKE"):
            #SNOWFlake INIT
            op,prev_op = self.OPINION,self.OPINION
            confidence_counter = 0
            counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]
            self.UNDECIDED = True
            majority = False
            while self.UNDECIDED:
                counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]
                majority = False
                self.get_query_opinion()
                if(self.QUERY_OPINION == 9512): #If the limit is exceeded then it will be triggered
                    self.OPINION = self.INITIAL_OPINION
                    sio.emit("consensus_decision",{"OPINION":self.OPINION})
                    break
                for x in self.QUERY_OPINION:
                    counter[x]+=1
                max_v = max(counter)
                if(max_v >= self.PROTOCOL_DETAILS["ALPHA"]):
                    majority = True
                    op = counter.index(max_v)

                if(op != prev_op):
                    prev_op = op
                    confidence_counter = 0
                else:
                    confidence_counter+=1

                if(majority == False):
                    confidence_counter = 0
                #print("OPINION,Q.OPINION,OP,PREV_op,CC,majo",self.OPINION,self.QUERY_OPINION,op,prev_op,confidence_counter,majority)
                if(confidence_counter >= self.PROTOCOL_DETAILS["BETA"]):
                    print("SENDING STUFF TO MANGAER")
                    self.OPINION = op
                    sio.emit("consensus_decision",{"OPINION":self.OPINION})
                    self.UNDECIDED = False
        
        if(self.PROTOCOL_DETAILS["NAME"] == "SNOWFLAKEPLUS"):
            #SNOWFlake INIT
            op,prev_op = self.OPINION,self.OPINION
            confidence_counter = 0
            counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]
            self.UNDECIDED = True
            majority = False
            while self.UNDECIDED:
                counter = [0 for x in range(self.PROTOCOL_DETAILS["NUM_CHOICES"])]
                majority = False
                self.get_query_opinion()
                if(self.QUERY_OPINION == 9512): #If the limit is exceeded then it will be triggered
                    self.OPINION = self.INITIAL_OPINION
                    sio.emit("consensus_decision",{"OPINION":self.OPINION})
                    break
                print("COUNTER:",counter,"QUERY_OPINOIN:",self.QUERY_OPINION)
                for x in self.QUERY_OPINION:
                    counter[x]+=1
                max_v = max(counter)
                if(max_v >= self.PROTOCOL_DETAILS["ALPHA"]):
                    majority = True
                    op = counter.index(max_v)

                if(op != prev_op):
                    prev_op = op
                    confidence_counter = 0
                else:
                    confidence_counter+=1

                if(majority == False or max_v < self.PROTOCOL_DETAILS["ALPHA2"]):
                    confidence_counter = 0
                #print("OPINION,Q.OPINION,OP,PREV_op,CC,majo",self.OPINION,self.QUERY_OPINION,op,prev_op,confidence_counter,majority)
                if(confidence_counter >= self.PROTOCOL_DETAILS["BETA"]):
                    print("SENDING STUFF TO MANGAER")
                    self.OPINION = op
                    sio.emit("consensus_decision",{"OPINION":self.OPINION})
                    self.UNDECIDED = False

                

    #Uploading Public to IPFS

class Expert:

    def __init__(self,NUMBER):
        with open("config.toml","rb") as f:
            self.CONFIG = load(f)
        with open(self.CONFIG["BOOTSTRAP"]["CONTRACT_ABI"],"rb") as f:
            self.ABI = jload(f)
        self.MALICIOUS = False
        print(NUMBER)
        if(self.CONFIG["MALICIOUS"]["LIST"][NUMBER] == 1):
            self.MALICIOUS = True
        self.NUMBER = NUMBER
        self.PREDICTIONS = [] #Holds the Opinion
        self.MODEL = Model(NUMBER%5,self.MALICIOUS)
        self.CONN = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
        self.CONN.eth.default_account = self.CONN.eth.accounts[NUMBER]
        self.PUB_ADDRESS = self.CONN.eth.accounts[NUMBER]
        print(self.PUB_ADDRESS) 
        self.CONTRACT = self.CONN.eth.contract(address=self.CONFIG["METADATA"]["CONTRACT_ADDRESS"],
                                          abi=self.ABI)
        

        if(self.CONFIG["BOOTSTRAP"]["NEW_KEY_GEN"]):
            self.PUBLICK_KEY_CID = self.new_key_generator("./keys/expert"+str(NUMBER)+"/")
        
        with open("./keys/expert"+str(NUMBER)+"/private.pem", 'rb') as pem_in:
            pemlines = pem_in.read()
        self.PRIVATE_KEY = load_pem_private_key(pemlines, None, default_backend()) 
        self.PUBLIC_KEY = self.PRIVATE_KEY.public_key()
        self.PADDING = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None)


        self.gather_data()#Gets data and Decrypts it and Creates File Objects
        self.MODEL.handle_data(self.DATA,self.TARGET) #Hands over data to Model Class
        temp = dt.now()
        self.MODEL.train_model(self.SAVE_PATH+"/expert"+str(self.NUMBER)+".pickle")
        self.MODEL_WEIGHT = self.MODEL.get_weight("./partitioned_data/validation.csv",self.TARGET)
        self.GAS_USED = 0
        print("MODEL WEIGHT:",self.MODEL_WEIGHT)
        print("[EXPERT] Training Done. Training Took:",dt.now()-temp) #Add Clock Here

    def new_key_generator(self,filepath):
        """
            Generate New Keys, save the private key and upload the publick key to IPFS
        """
        #Key Generation
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=1024, backend=default_backend())
        public_key = private_key.public_key()
        #Saving the Private Key in PEM file
        private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        with open(filepath+"private.pem", 'wb') as pem_out:
            pem_out.write(private_pem)
        with open(filepath+"public.pem", 'wb') as pem_out:
            pem_out.write(public_pem)
        hash = ipfs_api.http_client.add(filepath+"public.pem")
        hash = hash["Hash"]
        tra1 = self.CONTRACT.functions.updateExpertPublicKey(self.NUMBER,hash).transact()
        tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra1)
        self.GAS_USED += tx_receipt["gasUsed"]
        print("Generated New Key and Updated the Smart Contract",hash)


    ##HEre Goes Extra Code
    def gather_data(self):
        #Getting CID of Pickle file from Smart Contract
        data_cid = self.CONTRACT.functions.partition_cid(self.NUMBER).call()
        print("[EXPERT]GOT CID from Smart Contract for PICKLE RECORD:",data_cid)
        P_RECORD = ipfs_api.read(data_cid)
        P_RECORD = pload(P_RECORD)
        # Getting Key for FerNet Key in the picke
        KEY = self.PRIVATE_KEY.decrypt(P_RECORD['KEY_CIPHER'],self.PADDING)
        KEY = Fernet(KEY)
        D_RECORD = ipfs_api.read(P_RECORD["CID_DATA"])
        D_RECORD = KEY.decrypt(D_RECORD).decode()
        
        D_RECORD = StringIO(D_RECORD)
        
        self.DATA = D_RECORD
        #print(self.DATA.shape,"FOCUS $$$$$$$$$$$")
        self.TARGET,self.SAVE_PATH = P_RECORD["TARGET"],P_RECORD["SAVE_PATH"]
        print("[EXPERT]Decryption SUCCESFULL, DATA and TARGET Extracted")

temp = dt.now()
NUMBER = int(sys.argv[1])
EXPERT = Expert(NUMBER)
CONSENSUS = Consensus(1,10,EXPERT.CONFIG["PROTOCOL"])
sio.connect('http://localhost:45000')
print("MY OPINION:",OPINION)
print("[EXPERT]EXPERT IS READY. TIME TAKEN:",(dt.now()-temp).total_seconds())
sio.wait()
