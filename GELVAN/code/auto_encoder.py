import torch
import torch.nn as nn
import torch.optim as optim
from tomli import load
import sys
import pandas as pd
import ipfs_api
from cryptography.fernet import Fernet
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
import ipfs_api

#from ipfs_api import http_client as ips

import pandas as pd
import toml
class Encoder(nn.Module):
    def __init__(self, input_size):
        print("INPUT",input_size)
        super(Encoder, self).__init__()
        self.input_size = input_size
        with open("config.toml", "rb") as f:
            self.CONFIG = load(f)
            self.CONFIG = self.CONFIG["ENCODER"]

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 512),  # Input layer
            nn.LeakyReLU(),
            nn.Linear(512, 256),  # 1st hidden layer
            nn.LeakyReLU(),
            nn.Linear(256, 128),  # 2nd hidden layer
            nn.LeakyReLU(),
            nn.Linear(128, 64),  # 3rd hidden layer
            nn.LeakyReLU(),
            nn.Linear(64, self.CONFIG["EMBEDDING_LENGTH"])  # Embedding layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.CONFIG["EMBEDDING_LENGTH"], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.input_size)  # Final layer (linear activation for raw pixel values)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Embbeding_Generator:
    def __init__(self,ID):
        self.id = ID
        with open("config.toml","rb") as f:
                self.CONFIG = load(f)
                self.CONFIG = self.CONFIG["ENCODER"]
        self.CONN = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
        self.CONN.eth.default_account = self.CONN.eth.accounts[0]
        with open(self.CONFIG["CONTRACT_ABI"],"rb") as f:
            self.ABI = jload(f)
        self.CONTRACT = self.CONN.eth.contract(address=self.CONFIG["CONTRACT_ADDRESS"],
                                            abi=self.ABI)
        self.PADDING = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None)
        self.GAS_USED = 0
    def load_data(self):
        """
        - Loads the Data
        """
        data = pd.read_csv(self.CONFIG["BASE_PATH"] + str(self.id) + ".csv")
        y = data.pop(self.CONFIG["TARGET"])
        self.X_tensor = torch.tensor(data.values, dtype=torch.float)
        self.y_tensor = torch.tensor(y.values, dtype=torch.long)  # For classification, if needed
        self.input_size = data.shape[1]

    def train_model(self, batch_size=256):
        self.load_data()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move tensors to device
        self.X_tensor = self.X_tensor.to(device)
        self.model = Encoder(self.input_size).to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        
        # Create DataLoader for batch processing
        data_loader = torch.utils.data.DataLoader(self.X_tensor, batch_size=batch_size, shuffle=True)
        
        for epoch in range(self.CONFIG["EPOCHS"]):
            train_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                recon_x = self.model(batch)
                loss = loss_function(recon_x, batch)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            
            # Compute average loss
            train_loss /= len(data_loader)
            
            if epoch % 10 == 0:
                print(f'[DATA_PROVIDER {self.id}]Epoch [{epoch}] - Average Loss: {train_loss:.6f}')
                # import matplotlib.pyplot as plt

                # with torch.no_grad():
                #     sample = self.X_tensor[:10]
                #     recon_sample = self.model(sample).cpu().numpy()

                # fig, axs = plt.subplots(2, 10, figsize=(15, 3))
                # for i in range(10):
                #     axs[0, i].imshow(sample[i].view(28, 28).cpu().numpy(), cmap='gray')
                #     axs[1, i].imshow(recon_sample[i].reshape(28, 28), cmap='gray')
                #     axs[0, i].axis('off')
                #     axs[1, i].axis('off')
                # plt.show()
    def get_embeddings(self):
        input_data =self.X_tensor#CAREFULL HERE
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to compute gradients for inference
            embeddings = self.model.encoder(self.X_tensor)  # Get embeddings from the encoder
        data = pd.DataFrame(embeddings)
        data[self.CONFIG["TARGET"]] = self.y_tensor
        data.to_csv(self.CONFIG["BASE_PATH"]+"EMBEDDINGS"+str(self.id)+".csv",index=False)

    def save_stuff(self):
        mkdir("./encoder/encoder"+str(self.id))
        torch.save(self.model.encoder.state_dict(),"./encoder/encoder"+str(self.id)+"/auto_encoder_"+str(self.id)+".pth")
        torch.save(self.model.decoder.state_dict(),"./encoder/encoder"+str(self.id)+"/auto_decoder_"+str(self.id)+".pth")
        self.get_embeddings()
        copyfile("./raw_data/EMBEDDINGS"+str(self.id)+".csv","./encoder/encoder"+str(self.id)+"/EMBEDDINGS.csv")
        copyfile("./raw_data/"+str(self.id)+".csv","./encoder/encoder"+str(self.id)+"/RAW_DATA.csv")
        
        print("SAVED SUCCESFULY")
    def send_to_manager(self):
        """
            - Majorly Sends the CSV to Manager
                - To do that:
                    - Get Embeddings:
                        - Create a CSV
                        - Encrypt it  using f
                        - Save it as JSON
                        - UPLOAD TO IPFS
                        - Get CID 
                        - Encrypt the Resulting with KEY, CID and TARGET
                        - Create a Pickle/JSON
                        - Upload this to IPFS
                    - 
        """
        key = Fernet.generate_key()
        f = Fernet(key)
        with open(self.CONFIG["BASE_PATH"]+"EMBEDDINGS"+str(self.id)+".csv","rb") as file1:
            data = file1.read()
        enc = f.encrypt(data)#Need uploading
        with open(self.CONFIG["BASE_PATH"]+str(self.id)+".json","wb") as file1:
            file1.write(enc)
        cid = ipfs_api.http_client.add(self.CONFIG["BASE_PATH"]+str(self.id)+".json")
        cid = cid["Hash"]

        man_cid = self.CONTRACT.functions.expert_public_key(self.CONFIG["NUM_EXPERTS"]).call()
        print("EXPERT KEY IS:",man_cid)
        man_key = ipfs_api.read(man_cid) 
        man_key = load_pem_public_key(man_key)
        cipher = man_key.encrypt(key,self.PADDING)
        d={"KEY_CIPHER":cipher,"CID_DATA":cid,"TARGET":self.CONFIG["TARGET"]}
        with open(self.CONFIG["BASE_PATH"]+"part"+str(self.id)+".pickle", "wb") as write_file:
            pdump(d, write_file)

        cid = ipfs_api.http_client.add(self.CONFIG["BASE_PATH"]+"part"+str(self.id)+".pickle")
        cid = cid["Hash"]
        tra1 = self.CONTRACT.functions.updatePartition(self.id,cid).transact()
        tx_receipt = self.CONN.eth.wait_for_transaction_receipt(tra1)
        
        with open("./encoder/encoder"+str(self.id)+"/GAS_USAGE.csv","w") as f:
            f.write(str(tx_receipt["gasUsed"]))
        print(f'[DATA_PROVIDER {self.id}]Smart Contract Updated',cid,self.id)


"""
    To Do List:
        - Create a Wind Up Function that does the following :
            - Saves both Encoder and Decoder
            - Create a File By the Name DP_ID, which stores raw data, embeddings, and model
        - Need to figure out how to kick start the manager,
            - May be if ID == N, then it might create the flag
                - But there is a possiblity that rest all might be still working on it
"""

        
if (__name__ == "__main__"):
    a = Embbeding_Generator(int(sys.argv[1]))
    a.train_model()
    a.save_stuff()
    a.send_to_manager()

