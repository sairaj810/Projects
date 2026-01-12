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
from time import sleep
from pickle import loads as pload
from io import StringIO  
import ipfs_api


#from ipfs_api import http_client as ips
import eventlet
import socketio
import pandas as pd
import toml




sio = socketio.Server(logger=False,engineio_logger=False)
app = socketio.WSGIApp(sio)
sock = eventlet.listen(('', 45000))
MANAGER = ""
CONNECTED = []      #Holds the SID of the connected Experts
CONFIG = []         #Configuration from TOML
OPINIONS= {}        #A dict, which holds the opinions of the Experts, with SID as key
DECISION = {}       #A dict, with SID as key, expert's advice as a value , used for ensuring there is a consensus
REPEAT_COUNT = 0
PUBLIC_KEY,PRIVATE_KEY = 0,0
GAS_USED = 0


@sio.event
def connect(sid, environ):
    pass
    #print("[ATTEMPT] EXPERT TRIED CONNECTION")
    print(f"Expert {sid} connected")
    
@sio.event
def ready(sid):
    
    sio.enter_room(sid,"experts")
    CONNECTED.append(sid)
    print("[CONNECTION] NEW EXPERT CONNECTED. Waiting for",MANAGER.CONFIG["PROTOCOL"]["N"]-len(CONNECTED))
    MANAGER.ROUND_REPETATION_COUNTER[sid] = 0
    if(len(CONNECTED) == MANAGER.CONFIG["PROTOCOL"]["N"]):
        print("[CONSENSUS START] Starting the Consensus")##HEHE
        MANAGER.reputation_round()

@sio.event
def disconnect(sid):
    sio.leave_room(sid,"experts")
    #CONNECTED.remove(sid)
    print('[EXPERT DISCONNECTED] Disconnect ', sid)
    

@sio.event
def opinion_handler(sid,data):
    OPINIONS[data["for"]].append(data["opinion"])
    if(len(OPINIONS[data["for"]]) == data["k"]):
        sio.emit("query_opinion",to=data["for"],data = OPINIONS[data["for"]])
        OPINIONS[data["for"]] = []

@sio.event
def get_query(sid,data):
    #print("[QUERY]QUERY INSITIATED BY",sid)
    global OPINIONS
    k = data['query_size']#Requested K
    OPINIONS[sid] = []
    if(k > len(CONNECTED) - 1):
        print("[ERROR] Query Size Is Greater than Available Experts")
        sio.emit("handle_error","Query_Size Bigger than Number of Connected experts",to = sid)
        return
    track = [False for x in CONNECTED]
    choice,count = 0,0
    while count < k:
        choice = rand(0,len(CONNECTED)-1)
        if(CONNECTED[choice] != sid and not track[choice]):
            track[choice] = True
            count+=1
            sio.emit("give_opinion",{"for":sid,"k":k},to=CONNECTED[choice])
        #print("\t\t\t[FINDING ]")
    
    

@sio.event
def consensus_decision(sid,data):
    global DECISION, CONFIG,REPEAT_COUNT
    DECISION[sid] = data["OPINION"]
    if(len(DECISION) == MANAGER.CONFIG["PROTOCOL"]["N"]): #All Experts Advice Was Recieved
        s = set(tuple(DECISION.values()))
        MANAGER.ROUND_REPETATION_COUNTER[sid]+=1

        #print("FOCUS HERE:",MANAGER.ROUND_REPETATION_COUNTER)
        if(len(s)>1): #If Single Decree is not Found
            print("\t[SINGLE DECREE IS NOT FOUND] Starting Another Round")
            if( MANAGER.ROUND_REPETATION_COUNTER[sid] < 2):#2 Stands for Serverside LIMITING
                sio.emit("start_consensus",0,room="experts")
            else:#Creates the bak up mechanism
                print("\t\t\t\t Limit Execeeded, Triggering Back Up Mechanism")
                MANAGER.CONSENSUS_MESSED = True
                MANAGER.CONSENSUS_PREDICTED.append(DECISION[sid])
                DECISION = {}

        elif(len(s) == 1 ):
            print("\t[SINGLE DECREE FOUND] Opinion:",DECISION[sid])
            MANAGER.CONSENSUS_PREDICTED.append(DECISION[sid])
            DECISION = {}
         
       
        
@sio.event
def done(sid):
    global MANAGER
    if(not (sid in MANAGER.DONE)):
        MANAGER.DONE.append(sid)
    if(len(MANAGER.DONE) == MANAGER.CONFIG["PROTOCOL"]["N"]):
        MANAGER.DONE = []
        MANAGER.reputation_round()

@sio.event
def sc_done(sid):
    global MANAGER
    global GAS_USED
    
    if(not (sid in MANAGER.DONE)):
        MANAGER.DONE.append(sid)
    print("GOT ONE FOR SC DONE",len(CONNECTED),"WAITING FOR:",len(MANAGER.DONE))
    if(len(MANAGER.DONE) == MANAGER.CONFIG["PROTOCOL"]["N"]):
        #GRACEFULLY CLOSING
        weight = {}
        #Should Make if else here
        # if(MANAGER.CONSENSUS_MESSED):
        #     print("\t\t\t Generating Using BackUp Mechanism")
        #     tra1 = MANAGER.CONTRACT.functions.calculateBackUpTrust().transact()#LAST IS MANAGER.
        #     tx_receipt = MANAGER.CONN.eth.wait_for_transaction_receipt(tra1)
        #     GAS_USED += tx_receipt["gasUsed"]
        #     #Should Looop Here Around the CRETED STUFF, GET GAS PRICE AS WELL
        # else:
        tra0 = MANAGER.CONTRACT.functions.set_consensus(False).transact()#Setting up flag in SC
        print("[MANAGER] Setting the Flag in Smart Contract to",MANAGER.CONSENSUS_MESSED)
        tx_receipt = MANAGER.CONN.eth.wait_for_transaction_receipt(tra0)
        GAS_USED += tx_receipt["gasUsed"]
        tra1 = MANAGER.CONTRACT.functions.calculateReputation().transact()#LAST IS MANAGER.
        tx_receipt = MANAGER.CONN.eth.wait_for_transaction_receipt(tra1)
        GAS_USED += tx_receipt["gasUsed"]
        print("HOPEFULLYCALCULATED")
        expert_gas = []
        expert_weight = {} #Hold basic accuracy can be used as weight
        for x in range(0,MANAGER.CONFIG["PROTOCOL"]["N"]):
            w = MANAGER.CONTRACT.functions.finalscore(x).call()
            print(w)
            weight[x] = w
            with open(str(x)+".txt","r") as f:
                a = f.read()
                _,gas,c_weight = a.split(";")
                expert_gas.append(int(gas))
                expert_weight[x] = int(c_weight)
                rm(str(x)+".txt")
        
        FINAL_GAS_USED = {"MANAGER":GAS_USED,"EXPERT":expert_weight}
        FINAL_WEIGHT = {"TREEMA":weight,"ACCURACY":expert_weight,"BACKUP":MANAGER.CONSENSUS_MESSED}
        print("FINAL_WEIGHT:",FINAL_WEIGHT)
        print("FINAL_GAS:",FINAL_GAS_USED)

        with open(MANAGER.BASE+"/models/weights.pickle", "wb") as write_file:
            pdump(FINAL_WEIGHT, write_file)
        with open(MANAGER.BASE+"/models/gasUsed.pickle", "wb") as write_file:
            pdump(FINAL_GAS_USED, write_file)
        with open(MANAGER.BASE+"/models/CONFIG.toml", "w") as f:
                    toml.dump(MANAGER.CONFIG, f)
        print("CREATED WEIGHTS FOLDER")
        with open("./signal.flag", 'w') as f:
            f.write('')
        sio.emit("end")
        #sio.close_room("experts")
        #for x in CONNECTED:
            #sio.disconnect(x)
        
        sio.shutdown()
        
        exit(108)
        
        #SHUTDOWN
 
#######EVENTS GO UP:: FUNCTIONS GO DOWN




    


class Manager:
    def __init__(self):
        """
            Setup Initial Workings
            As of now:
                - Creates a new file named "keys" if new_key_gen is set to true 
                    - Create Sub Directories based on N
        Should Do:
            - YET TO THINK
        """
        #Creating file for storing this runs modesl
        global GAS_USED
        temp = len(listdir("./runs"))
        self.BASE = "./runs/attempt"+str(temp)
        mkdir(self.BASE)
        mkdir(self.BASE+"/models")
        mkdir(self.BASE+"/part_data")
        #READING STUFF
        with open("config.toml","rb") as f:
            self.CONFIG = load(f)
        with open(self.CONFIG["BOOTSTRAP"]["CONTRACT_ABI"],"rb") as f:
            self.ABI = jload(f)
        #SMART CONTRACT STUFF
        self.CONN = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
        self.CONN.eth.default_account = self.CONN.eth.accounts[0]
        self.CONTRACT = self.CONN.eth.contract(address=self.CONFIG["METADATA"]["CONTRACT_ADDRESS"],
                                            abi=self.ABI)
        print("COUNTER VALUE is:",self.CONTRACT.functions.counter().call())
        tra0 = self.CONTRACT.functions.setup(1,self.CONFIG["PROTOCOL"]["N"],self.CONFIG["PROTOCOL"]["N"],self.CONFIG["METADATA"]["MAX_ROUNDS"],).transact()
        tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra0)
        GAS_USED += tx_receipt["gasUsed"]
                
        #PUBLIC KEY STUFF
        self.EXPERT_PUBLIC_KEYS = {} #Holds the Public Keys of the Validator
        with open("./keys/manager/private.pem", 'rb') as pem_in:
            pemlines = pem_in.read()
        self.PRIVATE_KEY = load_pem_private_key(pemlines, None, default_backend()) 
        self.PUBLIC_KEY = self.PRIVATE_KEY.public_key()
        print("MANAGER PUBLIC",self.PUBLIC_KEY)
        self.PADDING = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None)
        #Uploading the Keys to IPFS and Setting up Smart Contract
        hash = ipfs_api.http_client.add("./keys/manager/public.pem")
        hash = hash["Hash"]
        tra1 = self.CONTRACT.functions.updateExpertPublicKey(self.CONFIG["PROTOCOL"]["N"],hash).transact()#LAST IS MANAGER.
        tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra1)
        GAS_USED += tx_receipt["gasUsed"]
        
        #Reading Public Keys 
        for x in range(0,self.CONFIG["PROTOCOL"]["N"]):
            cid = self.CONTRACT.functions.expert_public_key(x).call()
            print("Here is CID:",cid)
            #UNCOMMENT THESE
            key = ipfs_api.read(cid) 
            key = load_pem_public_key(key)
            self.EXPERT_PUBLIC_KEYS[x] = key
        print("[MANAGER] Read Public Keys")

        #NORMAL VARIABLE STUFF
        #self.TEST_DATA = pd.read_csv(self.CONFIG["METADATA"]["TEST_DATA_PATH"])
        self.MAX_REPUTATION_ROUND = self.CONFIG["METADATA"]["MAX_ROUNDS"] #Controls Max Number of Rounds, SHOULD BE FROM CINFIG
        self.ROUND_LIMIT = self.CONFIG["METADATA"]["ROUND_LIMIT"]
        self.current_reputation_round = 0
        self.DONE = []           #A listm which contains the managers who are done with consensus, used for maintaing concurrency
        self.GROUND_TRUTH = []   #A list which holds the predicted truth
        self.CONSENSUS_PREDICTED = []   #A list which holds the Consenus Truth
        self.ROUND_REPETATION_COUNTER = {} #Holds the # of rounds for processing current rounds
        self.CONSENSUS_MESSED = False #If false, then consensus did well

    def wait_for_Event(self):
        logs = []
        while len(logs) == 0:
            logs = self.CONTRACT.events.Embedding_Recieved().get_logs()
            print(logs)
            sleep(1)
        log = logs[-1]
        if(log["args"]["_value"] == 2 and log["event"] == "Embedding_Recieved"):
            print("Recieved")
        with open("dataready.flag", 'w') as f:
            f.write('')

    def data_preparation(self,NUM_PARITIONS,ENCRYPTION,IPFS_UPLOAD,UNIFORM,VARIENCE):

        global GAS_USED
        data = self.TRAIN_DATA
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

        #if(IPFS_UPLOAD == False):
        rmtree("partitioned_data",ignore_errors=True)
        mkdir("partitioned_data")
        #Basically creating data partition for each expert
        #Since it is uploaded to IPFS, it needed to be encrypted
        #Which need to be decoded by respective expert
        
        for y,x in enumerate(range(self.CONFIG["METADATA"]["NUM_PARTIONS"])):

            PARTITIONS[x].to_csv("partitioned_data/"+str(y)+".csv",index=False) #Creating CSVs            
            copyfile("partitioned_data/"+str(y)+".csv",self.BASE+"/part_data/"+str(y)+".csv") #Copying Them For Later Refernce
            if(y< self.CONFIG["PROTOCOL"]["N"]):
                key = Fernet.generate_key()
                f = Fernet(key)
                with open("partitioned_data/"+str(y)+".csv","rb") as file1:
                    data = file1.read()
                enc = f.encrypt(data)#Need uploading
                with open("partitioned_data/"+str(y)+".json","wb") as file1:
                    file1.write(enc)
                cid = ipfs_api.http_client.add("partitioned_data/"+str(y)+".json")
                cid = cid["Hash"]

                cipher = self.EXPERT_PUBLIC_KEYS[y].encrypt(key,self.PADDING)
                
                d={"KEY_CIPHER":cipher,"CID_DATA":cid,"TARGET":self.CONFIG["METADATA"]["TARGET"],"SAVE_PATH":self.BASE+"/models"}
                with open("./partitioned_data/part"+str(y)+".pickle", "wb") as write_file:
                    pdump(d, write_file)
                #print("HERE IS THE ERROR")
                cid = ipfs_api.http_client.add("./partitioned_data/part"+str(y)+".pickle")
                cid = cid["Hash"]
                tra1 = self.CONTRACT.functions.updatePartition(y,cid).transact()
                tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra1)
                GAS_USED += tx_receipt["gasUsed"]
            print("\t[MANAGER]Created Partition ",str(y)," and Updated SC with",cid)

            #TAKE THIS AND UPLOAD TO IPFS Update the CID to SMART CONTRACT
        PARTITIONS[-1].to_csv("partitioned_data/validation.csv",index=False) 
        self.TEST_DATA = PARTITIONS[-2]

        print("[MANAGER]PARTITION READY")

    def handle_data(self,data_cid):
        """
            Takes the CID
            Retirieve OG CID
            Decrypt it 
            Create Dataframe out of it
            return it
        """
        P_RECORD = ipfs_api.read(data_cid)
        P_RECORD = pload(P_RECORD)
        # with open("./keys/manager/private.pem", 'rb') as pem_in:
        #     pemlines = pem_in.read()
        # print("This is P:RECORD",P_RECORD)
        # self.PRIVATE_KEY = load_pem_private_key(pemlines, None, default_backend())
        KEY = self.PRIVATE_KEY.decrypt(P_RECORD["KEY_CIPHER"],self.PADDING)
        KEY = Fernet(KEY)
        D_RECORD = ipfs_api.read(P_RECORD["CID_DATA"])
        D_RECORD = KEY.decrypt(D_RECORD).decode()
        D_RECORD = StringIO(D_RECORD)
        dataframe = pd.read_csv(D_RECORD)
        
        return(dataframe)

    def retireve_data(self):
        limit = self.CONFIG["ENCODER"]["NUM_DATAPROVIDER"]
        data= []
        for x in range(0,limit):
            dataid = self.CONTRACT.functions.partition_cid(x+1).call()
            data.append(self.handle_data(dataid))#Returns Pandas Data Frame
        self.TRAIN_DATA = pd.concat(data)
        

        

    #Here
    def reputation_round(self):

        if(self.current_reputation_round >= self.MAX_REPUTATION_ROUND):
            #Handling the Clousore
            
            sio.emit("write_to_SC")#Just Write to Smart Contract
            self.update_smartcontract()
            #sio.close_room("nodes") #Closing The Nodes
            print("[DONE] PREDCITED VALUES:",self.CONSENSUS_PREDICTED)
            print("[DONE] ACTUAL VALUES:",self.GROUND_TRUTH)
            # for x in CONNECTED:
            #     sio.disconnect(x)
            #     sio.sleep(0.2)
            #sio.shutdown()#CAREFULL #######$$$$$s
            
            #exit()
            return
        #Going For One More Round
         
        self.current_reputation_round+=1
        
        #Need to write a code such that reputation is not happenig
        temp= self.TEST_DATA.iloc[rand(2,48)]#MAY BE SHOULD LOOK INTO THIS
        ground_truth = temp.pop(self.CONFIG["ENCODER"]["TARGET"])
        ground_truth = int(ground_truth)
        
        
        self.GROUND_TRUTH.append(ground_truth)
        print("[MANAGER] Starting the Next Round -> Real Vale:",self.GROUND_TRUTH[-1])
        #sleep(2)
        #Just Resettting the Counter
        for x in self.ROUND_REPETATION_COUNTER:
            self.ROUND_REPETATION_COUNTER[x] = 0
        sio.emit("start_consensus",{"Data":temp.tolist()},room="experts")

    def update_smartcontract(self):
        """
            - Update Both CONSENSUS PREDICTIONS and GROUNDTRUTH to SMART CONTRACT
        """
        global GAS_USED
        self.DONE = [] #Waiting For different Thing
        GT,CP ="",""
        for x in range(0,len(self.GROUND_TRUTH)):
            GT+=str(self.GROUND_TRUTH[x])
        for x in range(0,len(self.CONSENSUS_PREDICTED)):#DO Not OPTIMIZE IT
            CP+=str(self.CONSENSUS_PREDICTED[x])
        
        print("HERE ARE THE VALUES THAT SHOULD GO TO SC:",GT,CP)
        tra1 = self.CONTRACT.functions.updatePredictions(self.CONFIG["PROTOCOL"]["N"],GT).transact()
        tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra1)
        GAS_USED += tx_receipt["gasUsed"]
        tra1 = self.CONTRACT.functions.updatePredictions(self.CONFIG["PROTOCOL"]["N"]+1,CP).transact()
        tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra1)
        GAS_USED += tx_receipt["gasUsed"]
        print("COUNTER VALUE is:",self.CONTRACT.functions.counter().call())
        #GT=>N,CP=>N+1
        # is_calculated = False
        # while not is_calculated:  
        #     print("Waint")      
        #     is_calculated = self.CONTRACT.functions.calculated().call()
        # if(is_calculated):
        #     pass
        


t = dt.now()
MANAGER = Manager()
print("Created MANAGER")
MANAGER.wait_for_Event()
MANAGER.retireve_data()
#PATH,ENCRYPTION,IPFS_UPLOAD,UNIFORM
#NUM_PARTITIONS +1 is added so that extra partision can be created

MANAGER.data_preparation(MANAGER.CONFIG["METADATA"]["NUM_PARTIONS"]+2,#one for validation one for test data
                   MANAGER.CONFIG["METADATA"]["ENCRYPTION"],False,
                   MANAGER.CONFIG["METADATA"]["UNIFORM"],VARIENCE=50)
print("YOU MAY START MANAGER, Hopefully iT WOrk")
with open("./donepart.flag", 'w') as f:
     f.write('')
print("[MANAGER]SERVER STARTING: Time Took:",(dt.now()-t).total_seconds())
eventlet.wsgi.server(sock, app,log=None,log_output=False)
