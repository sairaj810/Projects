import os
import toml
from os import listdir
from os import rmdir,mkdir
from shutil import copytree,rmtree
from time import sleep
from random import randint as rand
def copy_all_files(filename):#WORK HERE
    #Ensure that PARTIONS WERE DELETE
    for x in os.listdir("./runs"):
        rmtree("./runs/"+x+"/part_data",ignore_errors=True)
    copytree("./runs/", "./results/"+filename, symlinks=False, ignore=None,)
    rmtree("runs",ignore_errors=True)
    mkdir("./runs")

def count_instance():#Used For Generating POPULAR ONE
     return len(os.listdir("./runs/results"))
    
def harvester(PROTOCOL):
    MAX_ITER = 5#Carefull
    for y in range(15,16,2):#EXPERTS CAREFUL 5 
        #CAREFULL 
        for z in range(14,15,3):#Round Numbers CAREFUL 5 
            current_iter = 0
            s = os.system("python3 setup.py "+str(y))#CAREFULL
            while current_iter < MAX_ITER:
                with open('config.toml', 'r') as f:
                    config = toml.load(f)
                    config["PROTOCOL"]["NAME"] = PROTOCOL
                    config["PROTOCOL"]["N"] = y
                    config["ENCODER"]["NUM_EXPERTS"] = y
                    config["METADATA"]["NUM_PARTIONS"] = y
                    config["METADATA"]["MAX_ROUNDS"] = z
                    config["MALICIOUS"]["PROP"] = 0
                    config["MALICIOUS"]["LIST"] = [0 for x in range(0,y)]
                with open('config.toml', 'w') as f:
                    toml.dump(config, f)
                s = os.system("./orchestrator.sh "+str(y-1))
                sleep(10)
                current_iter = count_instance()
                                
            copy_all_files(PROTOCOL+";NUM_EXPERT="+str(y)+";MAX_ROUNDS="+str(z))

rmtree("./results",ignore_errors=True)
os.mkdir("./results")
harvester("SNOWFLAKEPLUS")
