import numpy as np
import scipy as scp
from pyedflib import highlevel
import pyedflib as plib
import matplotlib.pyplot as plt
import mne as mne
import os

Subject = "001"
# signals, signal_headers, header = highlevel.read_edf("../dataset/files/S" + Subject + "/S" + Subject + "R" + Rep + ".edf")

path = "./dataset/files/S" + Subject + "/"

filenames = os.listdir(path)

filenames = [f for f in filenames if f.endswith(".edf")]
i = 0
tinit = 0.1
tfin = 2.1
all_epochs = []

# Iterar por todos os ficheiros do Subject
for f in filenames:   
    
    # Os primeiros dois eventos no dataset (primeiros dois ficheiros na folder) são baseline com olhos fechados e abertos, ignoro 
    if i > 1:
        raw = mne.io.read_raw(str(path + f))
        
        #   trials em que se faz mão esquerda vs direita (LR) e 2 mãos vs pés (UD)
        LR = {3, 4, 7, 8, 11, 12}
        UD = {5, 6, 9, 10, 13, 14}
        
        # Separar o encode que havia, para distinguir entre rest (1), mão esquerda (2), direita (3), ambas (4) ou pés (5); criar encode para as epochs
        if i+1 in LR:
            ids = {"T0": 0, "T1": 1, "T2": 2}
            epoch_ids = {"rest": 0, "left": 1, "right": 2}
        elif i+1 in UD:
            ids ={"T0": 0, "T1": 3, "T2": 4}
            epoch_ids = {"rest": 0, "hands": 3, "feet": 4}

        events = mne.events_from_annotations(raw, event_id = ids) 
        
        # Criar epochs nos dados, usando a chave estabelecida no if anterior
        # preload coloca as epochs na memoria
        # detrend serve para lidar com muscle artifacts 
        # TODO: aplicar isto ao sinal filtrado e não ao raw
        epoch = mne.Epochs(raw, events[0], epoch_ids, tinit, tfin, baseline = None, detrend = None, preload = True)
        
        epoch.plot(['C3..','C4..', 'Cz..'], 'auto', 3, block = True)
        all_epochs.append(epoch)
        
    
    i += 1

all_epochs = mne.concatenate_epochs(all_epochs)
print("AAAAAAAAAAA", np.shape(all_epochs.get_data()))
plt.plot(all_epochs.get_data()[:, 63, :])
# plt.figure() 
# plt.plot(all_epochs)


