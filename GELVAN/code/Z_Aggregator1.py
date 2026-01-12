import pandas as pd
from pickle import load
from os import listdir
class ResultProcessor:
    def __init__(self,base_path):
        self.base_path = base_path+"/results/"
        self.records = []
        self.results = []
        self.process()
    def process(self):
        for x in listdir(self.base_path):
            with open(self.base_path+x,"rb") as f:
                self.records.append(load(f))
        self.results = pd.DataFrame(self.records)
        self.results = self.results.mean().to_dict()
        print(self.base_path,x)
        parameters = self.base_path.split("/")[-3]
        print(parameters)
        stuff = parameters.split(";")
        self.results["PROTOCOL"] = stuff[0]
        self.results["NUM_EXPERTS"] = int(stuff[1].split("=")[-1])
        self.results["NUM_ROUNDS"] = int(stuff[2].split("=")[-1])
        self.results["EMBEDDING_LENGTH"] = int(self.results["EMBEDDING_LENGTH"])
        print(self.results)

class DirIterator:
    def __init__(self,base_path,name):
        self.holding = []
        self.base_path = base_path
        self.name = name
        self.final_result = []
        count = 0
        for x in listdir(self.base_path):
            self.holding.append(ResultProcessor(self.base_path+x))

        for x in self.holding:
            self.final_result.append(x.results)
        data = pd.DataFrame(self.final_result)
        data.to_csv(name,index=False)
        print("Created The CSV With Name:",name,"Happy Visualization")

a = DirIterator("./EXP2/","EXP2.csv")



