from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pickle import dump
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from random import randint as rand
from sklearn.metrics import accuracy_score
class Model:
    def __init__(self,choice,malicious):
        """
         This function is called inside the validator
         Since we are having 5 validators each should have its own stuff

         TWEAK THESE ACCORDINGLY TO CREATE HETROGENOUS MODELS
         for now let it be like this
         predict_proba should be set to true always

        Needed THings:
            - CHOICE: NUMBER OF VALIDATOR
            -DATA_CID: CID of the Paritioned Data
            -TARGET: Name of the target variable
            -HARD_VOTING: True or False
        """
        self.trained = False
        self.MALICIOUS = malicious
        print("FOCUS ON THIS NUMBER",choice)
        if(choice == 0):
            self.model = LogisticRegression(max_iter=100, solver='lbfgs')
            print("INside 0")
        elif(choice == 1):
            self.model = KNeighborsClassifier()
            print("INside 1")
        elif(choice == 2):
            self.model = SVC(probability=True)
            print("INside 2")
        elif(choice == 3):
            self.model = RandomForestClassifier(n_estimators = 100)
            print("INside 3")
        elif(choice == 4):
            self.model = RandomForestClassifier(n_estimators = 300, max_leaf_nodes=32)
            print("INside 4")
        #self.handle_data(choice,"lable")#TARGET goes here
        print("YO HERE IS MY MODEL:",self.model)

        
    def handle_data(self,DATA,TARGET):
        """
            Takes the data and create the ground truth vs feature
        Work To Do:
            - This method shall recieve the CID of the data that is parted
            - Get that CID from IPFS
                - Stuff is Encrypted; DEcrypt and 
        NOTE:
            - choice is temporary
        """
        #Get Data from IPFS
        #Decrypt it yep, using LONG DATA ENCRYPTION BASED RSA
        #Creating Stuff
        #   TEMPERORY ONLY
        self.data = pd.read_csv(DATA)
        self.target = self.data.pop(TARGET)

    def train_model(self,filepath):
        """
            Trains the model
            Should it be a seperate function(?) => Just Thinking Mostly not required we will see
            Few of the models may not have predict_proba as opition be careful
        """
        print("\t[MODEL_UTIL]Training Started...",self.data.shape)
        self.model.fit(self.data,self.target)
        self.trained = True
        with open(filepath,"wb") as f:
            dump(self.model,f)


        #Should Save the Model As Well
    
    def get_weight(self,filepath,target):
        data = pd.read_csv(filepath)
        target = data.pop(target)
        predicted = self.model.predict(data)
        return(round(100 * accuracy_score(target,predicted)))
        
    def predict(self,record):
        """
            Returns the output

            NOT MODEL AGNOSTIC BUT ONLY FOR MNIST DATASET
        """
        if(self.trained):

            data = pd.DataFrame(record)#Excluding the ground truth

            data = data.T#Renaming of Columsn Goes Here
            temp = self.model.predict_proba(data)
            #temp = self.model.predict_proba(pd.DataFrame(record[1:]).T)
            temp = temp[0].tolist()
            print("\t\t MODEL PREDICTION",temp.index(max(temp)))

            predicted = temp.index(max(temp))

            if(self.MALICIOUS):
                a = predicted
                a = int(not(bool(a)))#Since it is binary So Lets Be CareFul
                print("\t\t\t$$$$$$$$$$ORIGINAL is",predicted,"Giving is",a)
                return a
            return predicted

if __name__ == "__main__":
     t = Model(0)
     t.train_model()
     x = t.predict()
     for a in x:
        print(a,max(a))
