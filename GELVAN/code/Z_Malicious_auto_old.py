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
    MAX_ITER = 10
    mali = 0
    for y in range(11,13,2):#EXPERTS
        for z in range(5,6,3):#Round Numbers MODIFY THIS LOOP TO START WITH BEST VALUE, FOR NOW LETS STICK WITH 5
            for z1 in range(60,81,20):#MALICIOUS CAREFULL
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
                        gen_li = [0 for x in range(0,y)]
                        if z1 == 100:
                            gen_li = [1 for x in range(0,y)]
                        else:
                            mali = round((y*z1)*0.01) #Expert Malicious
                            temp = {}
                            while len(temp) < mali:
                                t1 = rand(0,y-1)
                                t = temp.get(t1,9512)
                                if(t==9512):
                                    temp[t1] = 1#1is Malicious
                            for x1 in temp:
                                gen_li[x1] = 1
                    config["MALICIOUS"]["LIST"] = gen_li
                    with open('config.toml', 'w') as f:
                        toml.dump(config, f)
                    s = os.system("./orchestrator.sh "+str(y-1))#CAREFULL
                    sleep(5)
                    current_iter = count_instance()


                copy_all_files(PROTOCOL+";NUM_EXPERT="+str(y)+";MAX_ROUNDS="+str(z)+";PROP="+str(z1))

def conformation():
    a = rand(5894,104597)
    print("You are GOING TO DELETE the EXISTING RESULTS,ENSURE YOU TAKE BACK UP")
   
    while True:
        b = int(input("IF YOU TOOK BACKUP TYPE:"+str(a)+":"))
        if b ==a :
            break
    
conformation()
rmtree("./results",ignore_errors=True)
os.mkdir("./results")
harvester("SNOWFLAKEPLUS")
