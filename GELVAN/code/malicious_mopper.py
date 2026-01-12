"""
- To Do This:
    - Create on single pickle
        - Load all the Ensemble i
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from os import listdir,mkdir
from pickle import load
from pickle import dump
from shutil import move
from shutil import copyfile,rmtree
from sklearn.ensemble import VotingClassifier
from tomli import load as load1
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import warnings
from multiprocessing import Process
warnings.filterwarnings("ignore")


class Ensemble:
    def __init__(self,filepath):
        self.MODELS = []
        self.WEIGHT = []
        self.BASE_PATH = filepath
        print("From the Ensenbme",filepath)
        for x in listdir(filepath+"/OLD/models"):

            if(x[0]=="e"):#Loding the models starting with e
                with open(filepath+"/OLD/models/"+x,"rb") as f:
                    self.MODELS.append(load(f))
            elif(x[0]=="w"):#loading the weights
                with open(filepath+"/OLD/models/"+x,"rb") as f:
                   self.WEIGHT = (load(f))
                    
            elif(x[0] == "g"):
                with open(filepath+"/OLD/models/"+x,"rb") as f:
                   self.GAS_USAGE = (load(f))

        
        with open(filepath+"/CONFIG.toml","rb") as f:
            self.config = load1(f)
        self.MALICIOUS  = self.config["MALICIOUS"]["LIST"]
        self.NUM_CHOICES = self.config["PROTOCOL"]["NUM_CHOICES"]
        self.CONFIG = self.config["ENCODER"]
        #Just Taking the Accuracy Weights
        self.ACCURACY_WEIGHT = []
        temp = self.WEIGHT["ACCURACY"]
       
        for x in range(0,len(temp)):
            self.ACCURACY_WEIGHT.append(temp[x])
        t = sum(self.ACCURACY_WEIGHT)
        self.ACCURACY_WEIGHT = [x/t for x in self.ACCURACY_WEIGHT]
        self.ACCURACY_WEIGHT = np.array(self.ACCURACY_WEIGHT)
        
        #Just Taking the TREEMA Weights
        self.TREEMA_WEIGHT = []
        temp = self.WEIGHT["TREEMA"]
        #print("\tSAIRAM:")
        #print("\t\tMALICIOUS:",self.MALICIOUS)
        #print("\t\tAccurayc:",self.ACCURACY_WEIGHT)
        #print("\t\tReputation:",temp)

        for x in range(0,len(temp)):
            self.TREEMA_WEIGHT.append(temp[x])
        t = sum(self.TREEMA_WEIGHT)
        self.TREEMA_WEIGHT = [x/t for x in self.TREEMA_WEIGHT]
        self.TREEMA_WEIGHT = np.array(self.TREEMA_WEIGHT)

        


    def set_model(self,data_path,target,encoder_path):
        """
            - Use the case accordingly
        """
        #Loading the Model
        data = pd.read_csv(data_path)
        y_val = data.pop(target)
        data = torch.tensor(data.values,dtype=torch.float32)
        encoder = nn.Sequential(
                    nn.Linear(data.shape[1], 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64,self.CONFIG["EMBEDDING_LENGTH"]))#Should be config.toml

        encoder.load_state_dict(torch.load(encoder_path))
        encoder.eval()
        with torch.no_grad():  # No need to compute gradients for inference
                embeddings = encoder(data)
        self.TEST_DATA = pd.DataFrame(embeddings)
        self.TEST_DATA_LABLES = y_val

        
    
   

    def predict_accu(self,voting = "hard",init = False):
        #This is classical Ensemble Learning where you should use the 
        #ACCURACY BASED STUFF
        
        if(voting == "hard"):
            pred = []
            final_pred = []
            temp = []
            for i,x in enumerate(self.MODELS):#Poulating the Predictions
                pred.append(x.predict(self.TEST_DATA))
            pred = np.array(pred)
            for i,x in enumerate(self.MALICIOUS):
                if (x==1):
                    pred[i] = pred[i]+1
            pred = np.transpose(pred)
            unique = [x for x in range(0,self.NUM_CHOICES)]
            final_pred = []
            for i in range(pred.shape[0]):
                weighted_votes = np.zeros(10)#Just Initialzing it to zero
                for j,p in enumerate(unique):
                    weighted_votes[j] = np.sum(self.ACCURACY_WEIGHT[ pred[i] == p])
                final_pred.append(unique[np.argmax(weighted_votes)])
                
            t = accuracy_score(self.TEST_DATA_LABLES,final_pred)
            return(t)

        if(voting == "soft"):
            pred= {}
            DATA_FRAMES = []
            for i,x in enumerate(self.MODELS):
                pred[i] = x.predict_proba(self.TEST_DATA)

            for i,x in enumerate(self.MALICIOUS):
                if(x == 1):#MALICIOUS
                    max_indices = np.argmax(pred[i], axis=1)
                    row_indices = np.arange(pred[i].shape[0])
                    pred[i][row_indices, max_indices] = 0

            final_result = 0
            for i,x in enumerate(pred):
                d = pd.DataFrame(pred[x])
                d = d*self.ACCURACY_WEIGHT[i]
                if(i == 0):
                    final_result = pd.DataFrame(0,index = d.index,columns=d.columns)
                final_result = final_result+d         
            
            final_pred = final_result.idxmax(axis=1).to_list()
            t = accuracy_score(self.TEST_DATA_LABLES,final_pred)
            return(t)

    def predict_rept(self,voting = "hard",init = False):
        
        unique = [x for x in range(0,self.NUM_CHOICES)]
        final_pred = []
        pred = []

        if(voting == "hard"):
            temp = []
            for i,x in enumerate(self.MODELS):#Poulating the Predictions
                pred.append(x.predict(self.TEST_DATA))
            pred = np.array(pred)
            #Introfucing the Maliciousness
            for i,x in enumerate(self.MALICIOUS):
                if (x==1):
                    pred[i] = pred[i]+1
            pred = np.transpose(pred)
            for i in range(pred.shape[0]):
                weighted_votes = np.zeros(10)#Just Initialzing it to zero
                for j,p in enumerate(unique):
                    weighted_votes[j] = np.sum(self.TREEMA_WEIGHT[ pred[i] == p])
                final_pred.append(unique[np.argmax(weighted_votes)])
                
            t = accuracy_score(self.TEST_DATA_LABLES,final_pred)
            return(t)

        if(voting == "soft"):
            pred= {}
            DATA_FRAMES = []
            for i,x in enumerate(self.MODELS):
                pred[i] = x.predict_proba(self.TEST_DATA)
            for i,x in enumerate(self.MALICIOUS):
                if(x == 1):#MALICIOUS
                    max_indices = np.argmax(pred[i], axis=1)
                    row_indices = np.arange(pred[i].shape[0])
                    pred[i][row_indices, max_indices] = 0
            final_result = 0
            for i,x in enumerate(pred):
                d = pd.DataFrame(pred[x])
                d = d*self.TREEMA_WEIGHT[i]
                if(i == 0):
                    final_result = pd.DataFrame(0,index = d.index,columns=d.columns)
                final_result = final_result+d         
            
            final_pred = final_result.idxmax(axis=1).to_list()
            t = accuracy_score(self.TEST_DATA_LABLES,final_pred)
            return(t)

        
def processor(ENCODER_NUMBER,to_process,destination):
    """
        - Process the Auto Encoder
    """
    
    a = Ensemble(to_process)
    a.set_model("./input_data/test.csv",a.CONFIG["TARGET"],destination+"/encoder/encoder"+ENCODER_NUMBER+"/auto_encoder_"+ENCODER_NUMBER+".pth")
    HARD_REPT = a.predict_rept()
    SOFT_REPT = a.predict_rept("soft")
    HARD_ACCU = a.predict_accu()
    SOFT_ACCU = a.predict_accu("soft")
    MANAGER_GAS = a.GAS_USAGE["MANAGER"]
    temp = list(a.GAS_USAGE["EXPERT"].values())
    EXPERT_GAS = sum(temp)/len(temp)

    with open(destination+"/ENSEMBLE_"+ENCODER_NUMBER+".pickle","wb") as f:
        dump(a,f)#Need TIME HERE
    return({"HARD_REPT":HARD_REPT,"SOFT_REPT":SOFT_REPT,
    "HARD_ACCU":HARD_ACCU,"SOFT_ACCU":SOFT_ACCU,"MANAGER_GAS":MANAGER_GAS,"EXPERT_GAS":EXPERT_GAS})

class Parent_Direct:
    def __init__(self,base_path):
        self.BASE_PATH = base_path
        rmtree(self.BASE_PATH+"MALICIOUS_RESULT",ignore_errors=True)
        mkdir(self.BASE_PATH+"MALICIOUS_RESULT")


    def repeat_caller(self):
        for x in listdir(self.BASE_PATH):
            if(x[0] == "r" or x[0] == "M"):
                continue
            a1 = self.actual_caller(self.BASE_PATH+x,"1")
            b1 = self.actual_caller(self.BASE_PATH+x,"2")#Careful
            c1 = (a1+b1)/2
            c1 = c1.tolist()

            final_result = {"HARD_ACCU":c1[0],"HARD_REPT":c1[1],"SOFT_ACCU":c1[2],
                            "SOFT_REPT":c1[3],"MANAGER_GAS":c1[4],"EXPERT_GAS":c1[5]}
            with open(self.BASE_PATH+"MALICIOUS_RESULT/"+x+".pickle","wb") as f:
                dump(final_result,f)

           #break #Careful

    def actual_caller(self,path,ENCODER_NUMBER):
        a = Ensemble(path)
        a.set_model("./input_data/test.csv",a.CONFIG["TARGET"],a.BASE_PATH+"/encoder/encoder"+ENCODER_NUMBER+"/auto_encoder_"+ENCODER_NUMBER+".pth")
        HARD_REPT = a.predict_rept()
        HARD_ACCU = a.predict_accu()
        SOFT_REPT = a.predict_rept("soft")
        SOFT_ACCU = a.predict_accu("soft")
        MANAGER_GAS = a.GAS_USAGE["MANAGER"]
        temp = list(a.GAS_USAGE["EXPERT"].values())
        EXPERT_GAS = sum(temp)/len(temp)
        return np.array([HARD_ACCU,HARD_REPT,SOFT_ACCU,SOFT_REPT,MANAGER_GAS,EXPERT_GAS])

base_path = "./EXP3"
procs = []
count = 0
for x in listdir(base_path):
    a = Parent_Direct(base_path+"/"+x+"/")
    proc = Process(target = a.repeat_caller)
    procs.append(proc)
    proc.start()
    count+=1



for proc in procs:
    proc.join()
print("All Process Done/ Run the Malicious COde and Make it WOrk")

