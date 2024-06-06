import numpy as np
import scipy as scp
import random
from pyedflib import highlevel
import pyedflib as plib
import matplotlib.pyplot as plt
import mne as mne
from scipy import linalg as la
from sklearn.linear_model import Perceptron
from sklearn.cluster import AffinityPropagation
import os
import pydirectinput

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle

import os

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, datafiles = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                datafiles.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        datafiles.extend(f)
    return subfolders, datafiles


subfolders, datafiles = run_fast_scandir('./dataset/files', [".edf"])

subfolders.sort()
datafiles.sort()
print(subfolders[:3])
print(datafiles[:3])


min_freq = 4
max_freq = 40

n_bands = int((max_freq - min_freq)/2)

bands = []
f1 = 4
f2 = 8

for i in range(1,n_bands):
  bands.append([f1,f2])
  f1+=2
  f2+=2
print("Bandas de frequência: ", bands)

mne.set_log_level('INFO')

filter_design = 'firwin2'
ska = 'edge'

raw = mne.io.read_raw_edf(datafiles[0], preload = False)
channels = raw.ch_names
n_channels = len(channels)

for band in bands:

  filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'],l_freq=band[0], h_freq=band[1], fir_design = filter_design) # Criar filtro. É preciso dados como parâmetro
  #mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.5, 60)) # Plot do filtro

mne.set_log_level('WARNING')

def check_event_count(event_data):

  '''
  Função que:
  - Faz prints do número de trials/blocos de cada classe de movimento. 

  ---------------------
  Input:
  - event_data: Objeto com todos os blocos de dados

  '''

  print("Rest: ", event_data["rest"].__len__())
  print("Right Fist: ", event_data["right_fist"].__len__())
  print("Left Fist: ", event_data["left_fist"].__len__())
  print("Both Fists: ", event_data["both_fists"].__len__())
  print("Both Feet: ", event_data["both_feet"].__len__())
     

''' Commit "Test/Train split: 84144ec" tem todo o código com os prints e plots desta função relacionados com os eventos'''

def load_subject_data(subject, datafiles, bands, tstart, tstop):

  '''
  Função que:
  - Carrega todos os dados contínuos de um participante da base de dados.
  - Filtra os dados contínuos.
  - Reatribui as labels para separar melhor as 5 classes de EEG da base de dados.
  - Agrupa e concatena os blocos dados dos vários trials (bloco com cerca de 4.2s de dados que 
  corresponde à execução de uma classe de movimento) de ficheiros 
  diferentes a partir das novas labels.
  - Segmenta os dados 

  ---------------------
  Input:
  subject: Caminho para o diretório onde estão todos os ficheiros de um participante 
  datafiles: Caminhos de todos os ficheiros edf (todos os participantes) da base de dados
  tstart: Instante no tempo (s) da primeira amostra do segmento de um bloco a ser analisado
  tstop: Instante no tempo (s) da última amostra do segmento de um bloco a ser analisado
  
  ---------------------
  Output:
  subject_epochs: Objeto com todos os blocos de dados concatenados e com toda a informção relativamente
  aos eventos de cada classe
  '''

  trials = []
  for trial in datafiles: 
    if subject in trial: 
      trials.append(trial)

  eachBand_epochs = []
  
  for band in bands: # Filtrar os dados nas várias bandas de frequência
    print(f"Filtragem na banda de frequências: {band[0]} - {band[1]} Hz")

    all_trials = [] # Variável com todos os blocos
    
    for count, trial  in enumerate(trials):
      if count > 1: # Avança os dois primeiros trials correspondentes ao rest
        raw_data = mne.io.read_raw_edf(trial, preload = True) # Dados EEG para um trial específico. 'Preload=True' serve para colocar o ficheiro na memória e não apenas referenciá-lo no diretório
        
        ch_names = raw_data.ch_names # Elétrodos
        fs =  raw_data.info.get('sfreq') # Frequência de amostragem
        annotations = raw_data.annotations # Variável que contém os dados no tempo relativos aos blocos de movimentos executados

        trial_type1 = [3, 4, 7, 8, 11, 12]
        trial_type2 = [5, 6, 9, 10, 13, 14]

        # Atribuição das labels corretas aos eventos de cada trial
        #--------------- TOMÁS: tive de adaptar a linha seguinte por causa da maneira como tenho o diretório (../ yada yada, cria uma separação para cada ponto,
        # então fica algo tipo [''] [''] ['actual diretorio'] ['edf']) -----------------------------------------
        trial_number = int(trial.split('.')[2][-2:]) # De 1 a 14; [0] corresponde à primeira parte da string original;
        
        if trial_number in trial_type1:
          event_ids = {'T0': 1, 'T1': 2, 'T2': 3} # Dicionário com os ids dos eventos de acordo com a descrição a laranja ^^
          epoch_event_ids = {"rest/block": 1, "left_fist/block": 2, "right_fist/block": 3}
        elif trial_number in trial_type2:
          event_ids = {'T0': 1, 'T1': 4, 'T2': 5}
          epoch_event_ids = {"rest/block": 1, "both_fists/block": 4,"both_feet/block": 5}
        else:
          event_ids = {'T0': 1} # Dois primeiros trials de EEG

        # Filtragem

        filtered_data = raw_data.copy().filter(band[0], band[1], fir_design=filter_design, skip_by_annotation=ska) 
        
        # Mapeamento dos eventos com as novas labels

        events = mne.events_from_annotations(filtered_data, event_id = event_ids) # Variável eventos: Retorna 3 colunas [Indice da amostra onde começa o bloco, Trigger, Tipo de movimento (label)]
        
        # Agrupamento dos blocos do mesmo id (epoching)

        epochs = mne.Epochs(filtered_data, events[0], event_id = epoch_event_ids, tmin = tstart, tmax = tstop, detrend = 1, baseline = None, preload = True)
        
        all_trials.append(epochs)
        
    # Concatenação de todos os blocos (trials) de uma banda de frequências

    subject_epochs = mne.concatenate_epochs(all_trials, add_offset = True, on_mismatch = 'raise')
    equalized_subject_epochs = subject_epochs.equalize_event_counts()[0]

    eachBand_epochs.append(equalized_subject_epochs)
    
  return eachBand_epochs

# Carregar dados de todos os participantes
# Equaliza o nº de trials em todas as classes. A classe rest tinha muitos mais trials por exemplo.
# O bom balanceamento de classes é importante durante a classificação

mne.set_log_level('CRITICAL')

all_subject_epochs = []

LOAD_FITLERED_FROM_FILES = False

if LOAD_FITLERED_FROM_FILES:
  for id, subject in enumerate([subfolders[41]]):
    print("---------------------------------------------")
    print("Ler dados do participante: ", id+1)
    with open(".\\filtered_data\\saved_file"+subject.split('\\')[-1]+".pkl", 'rb') as f:
      all_subject_epochs.append(pickle.load(f))


else:
  for id, subject in enumerate([subfolders[41]]):
    print("---------------------------------------------")
    print("Ler dados do participante: ", id+1)
    folder_data = load_subject_data(subject, datafiles, bands, 0.5, 3)
    with open(".\\filtered_data\\saved_file"+subject.split('\\')[-1]+".pkl", 'wb') as f:
      pickle.dump(folder_data, f)

    all_subject_epochs.append(folder_data)

  mne.set_log_level('WARNING')

subject_epochs = all_subject_epochs[0][0]

print("\nEqualização de eventos: ")
check_event_count(subject_epochs)
print("\n")

# Seleção de elétrodos
print(channels)
chan = mne.pick_channels(channels, include = ['F3..', 'Fz..','F4..', 
                                              'Fc3.', 'Fcz.','Fc4.',
                                                'C3..','Cz..','C4..',
                                                'Cp3.', 'Cpz.', 'Cp4.',
                                                'P3..', 'Pz..', 'P4..',
                                                'Oz..' ])
print("Indices: ", chan)

#dir(subject_epochs)
print("Condições: ",subject_epochs.event_id)
print("\nBlocos: ",subject_epochs.events)
subject_epochs["right_fist"].plot_image(picks=[chan[0]])


from sklearn.model_selection import train_test_split

def split_EEG(subject_EEG, classes, percent_train=0.8):
  
  n_trials, _, _ = np.shape(subject_EEG)
  n_train = round((n_trials/len(classes)) * percent_train)
  n_test = int((n_trials/len(classes)) - n_train)

  # print("Nº de trials: ", n_trials)
  # print("Nº de trials para treinar (por classe): ", n_train)
  # print("Nº de trials para teste (por classe): ", n_test)

  epochs_train = []
  epochs_test = []

  #shuffle trials. O miguel nao fazia. N sei se é sposto n fazer. o on missmatch com raise avisa que n esta por ordem cronologica
  np.random.seed(42)
  ids = np.random.permutation(n_trials//len(classes))
  ids_train = ids[:n_train]
  ids_test = ids[n_train:n_train+n_test]

  for class_key, _ in classes.items():
    epochs_train.append(subject_EEG[class_key][ids_train]._data)
    epochs_test.append(subject_EEG[class_key][ids_test]._data)

  #conver to numpy array
  epochs_train = np.array(epochs_train)
  epochs_test = np.array(epochs_test)

  #filter only channels of interest
  epochs_train = epochs_train[:, :, chan, :]
  epochs_test = epochs_test[:, :, chan, :]
  
  print("Shape dos dados de treino: ", epochs_train.shape)
  print("Shape dos dados de teste: ", epochs_test.shape)

  return epochs_train, epochs_test

# Split Test/Train

classes = subject_epochs.event_id

all_train_data = []
all_test_data = []

for subject in range(len(all_subject_epochs)):

  bands_train_data = []
  bands_test_data = []
  print("Sujeito: ", subject+1)

  for band in range(len(bands)):
    train_data, test_data = split_EEG(all_subject_epochs[subject][band], classes, percent_train=0.8)
    bands_train_data.append(train_data)
    bands_test_data.append(test_data)

  all_train_data.append(bands_train_data)
  all_test_data.append(bands_test_data)

# all_train_data = np.array(all_train_data)
# all_test_data = np.array(all_test_data)

# print("Shape final dos dados de treino: ", all_train_data.shape)
# print("Shape final dos dados de teste: ", all_test_data.shape)

import matplotlib.pyplot as plt
colors = ['b', 'r', 'g', 'y', 'm']

subject = 0
print("Sujeito: ", subject+1)

for band in range(len(bands)):
    break
    print("Banda: ", band+1)
    plt.figure(figsize=(10, 5))
    for class_ in range(5):
        # for i in range(all_train_data[subject][band][0][0].shape[0]): 
        #     plt.plot(all_train_data[subject][band][0][0][i])
        plt.scatter(all_train_data[subject][band][class_][0][0], all_train_data[subject][band][class_][0][2], color = colors[class_])
    plt.title(f"EEG data - Sujeito {subject+1} - Banda {band+1}")
    plt.show()


def train_CSP(X, Y, num_filters):
    X -= np.mean(X, axis=2)[:, :, np.newaxis]

    num_classes = len(np.unique(Y))

    covariances = []
    for i in range(num_classes):
        X_class = X[Y == i]
        S_class = np.mean([np.matmul(X_class[j], X_class[j].T) for j in range(X_class.shape[0])], axis=0)
        covariances.append(S_class)

    S0 = covariances[0]
    S1 = covariances[1]

    d, V = la.eigh(S0, S0 + S1)

    # get order of eigenvalues
    idx = np.argsort(np.abs(d - 0.5))[::-1]

    # reorder the eigenvectors
    V = V[:, idx]

    # transpose
    W = V.T

    # compute the patterns
    pattern = np.linalg.pinv(V)

    #select the two most important filters
    W = W[: num_filters]

    return W


def apply_csp(X, W, mean_power=False):
    X_csp = np.asarray([np.dot(W, epoch) for epoch in X])
    if mean_power:
        X_csp = (X_csp**2).mean(axis=2)
    
    return X_csp


def get_filters(X_input, y, J=6):
    filters = []

    for chosen_band in range(len(bands)):
        band_filters = []
        for chosen_class in range(5):
            X = X_input[chosen_band]
            X = X.reshape(-1, X.shape[2], X.shape[3])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

            y_train = (y_train == chosen_class).astype(int)
            y_test = (y_test == chosen_class).astype(int)

            W = train_CSP(X_train, y_train, J)

            band_filters.append(W)
        filters.append(band_filters)

    return filters


from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

def get_top_bands(X, filters, n_top_bands):
    all_class_filters = []

    for chosen_class in range(5):
        all_filters = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        X_test = X_test.transpose(1, 0, 2, 3)

        X_train = X_train.transpose(1, 2, 3, 0)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2] * X_train.shape[3])

        for band in range(len(bands)):
            for filter in range(len(filters[band])):
                X_csp = X_train[band]
                X_csp = apply_csp(X_train[band], filters[filter][chosen_class])
                all_filters.append(X_csp[0])

        all_filters = np.array(all_filters)
        all_filters = all_filters.transpose(0, 2, 1).reshape(all_filters.shape[0], all_filters.shape[2] * all_filters.shape[1])
        clustering = AffinityPropagation(random_state = 5).fit(all_filters)
        #clustering = KMeans(n_clusters=8, random_state=0).fit(all_filters)
        #print(chosen_class, "LEN", len(clustering.cluster_centers_))

        # Ele escolheu os centros (que fazem parte do all_filters que lhe demos), agora  é preciso identificar qual é qual 

        selected_filters = []
        for center in clustering.cluster_centers_:
            min_dist = float('inf')
            selected_filter = 0
            for i, filter in enumerate(all_filters):
                dist = np.linalg.norm(center - filter)
                if dist < min_dist:
                    min_dist = dist
                    selected_filter = i
            selected_filters.append(selected_filter)

        selected_filters = [[(f//len(filters[band])), (f%len(filters[band]))] for f in selected_filters]
        selected_filters = np.array(selected_filters)
        all_class_filters.append(selected_filters)

    band_counters = np.zeros((len(bands), 5))

    for class_ in range(5):
        for band, filter in all_class_filters[class_]:
            band_counters[band, class_] += 1

    band_counters_ranked = np.sum(band_counters, axis=1)

    # plt.plot(band_counters_ranked)
    # plt.show()

    return np.argsort(band_counters_ranked)[::-1][:n_top_bands]

# The bands and number of features were chosen based on the bands that resulted in less redundant information. These were determined using Affinity Propagation and K_means.
# 
# After experimenting, we concluded that band 5 was crucial, and bands 9,12,13,14 were very important, but somewhat redundant among themselves.

# After trying various classifiers, we came to the conclusion that the best was the Support Vector Machine with "poly" kernel. Although some Multi-Layered Perceptron architectures achieved simmalar results, these were always in a simmilar same range as the Support Vector Machine, never better. The SVM is a simpler model and as such is prefered, when the accuracy is the same.

def train_classifier(X, y, chosen_bands, J):
    kernel_type = ["rbf", "linear", "poly", "sigmoid"]
    models = []
    input_signal = []
    score_list = []
    filter_banks = []
    

    for band in chosen_bands:
        signal = X[:, band, :, :]

        #trian test split
        X_train, X_test, y_train, y_test = train_test_split(signal, y, test_size=0.3, random_state=42)


        W_bank = []
        for i in range(5):
            W = train_CSP(X_train, 1.*(y_train==i), J)
            W_bank.append(W)
        filter_banks.append(W_bank)
        

        for W in W_bank:
            X_csp = apply_csp(X_test, W)
            power = np.log(np.std(X_csp, axis=2))
            # power = np.std(X_csp, axis=2)
            input_signal.append(power)


    input_signal = np.array(input_signal)
    input_signal = input_signal.transpose(1, 0, 2)
    input_signal = input_signal.reshape(input_signal.shape[0], input_signal.shape[1]*input_signal.shape[2])


    #train lda
    for k in kernel_type:
        clf = SVC(kernel=k)
        print("\n\n", k)
        scores = cross_val_score(clf, input_signal, y_test, cv=2, scoring='balanced_accuracy')
        print(scores)
        print("mean", np.mean(scores))
        #print("std", np.std(scores))
        score_list.append(np.mean(scores))
        clf.fit(input_signal, y_test)
        models.append(clf)

    best_kernel = kernel_type[np.argmax(score_list)]
    print("Best kernel: ", best_kernel)

    return models[np.argmax(score_list)], filter_banks, np.max(score_list)
    
def preprocess(X, chosen_bands, filter_banks):
    input_signal = []


    for idx, band in enumerate(chosen_bands):
        signal = X[:, band, :, :]

        for W in filter_banks[idx]:
            X_csp = apply_csp(signal, W)
            power = np.log(np.std(X_csp, axis=2))
            # power = np.std(X_csp, axis=2)
            input_signal.append(power)


    input_signal = np.array(input_signal)
    input_signal = input_signal.transpose(1, 0, 2)
    input_signal = input_signal.reshape(input_signal.shape[0], input_signal.shape[1]*input_signal.shape[2])

    return input_signal


def classify(X_processed, model):
    return model.predict(X_processed)

def inference(X, chosen_bands, filter_banks, model):
    X_processed = preprocess(X, chosen_bands, filter_banks)
    return classify(X_processed, model)


jumento_celestino = []

for subject in range(len(all_train_data)):
    X = np.array(all_train_data[subject])
    y = np.zeros(X.shape[1]* X.shape[2])
    for class_ in range(X.shape[1]):
        y[class_ * X.shape[2]: (class_ + 1) * X.shape[2]] = class_

    filters = get_filters(X, y)

    X = X.transpose(1, 2, 0, 3, 4)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4])

    n_features = 3
    J = 6

    chosen_bands = get_top_bands(X, filters, n_features)

    print("Chosen Bands:", chosen_bands)

    model, filter_banks, validation_acc = train_classifier(X, y, chosen_bands, J)

    #test with X and y
    X_test = np.array(all_test_data[subject])
    X_test = X_test.transpose(1, 2, 0, 3, 4)
    X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2], X_test.shape[3], X_test.shape[4])

    y_test = np.zeros(X_test.shape[0])
    for i in range(5):
        y_test[i*X_test.shape[0]//5:(i+1)*X_test.shape[0]//5] = i

    print("Shapes: ", X_test.shape, y_test.shape)

    print(X_test.shape, y_test.shape)

    y_pred = inference(X_test, chosen_bands, filter_banks, model)
    print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))

    jumento_celestino.append([balanced_accuracy_score(y_test, y_pred), validation_acc])

jumento_celestino = np.array(jumento_celestino)


bar_width = 0.35
index = np.arange(len(jumento_celestino))

# plt.figure(figsize=(10, 5))
# plt.bar(index, jumento_celestino[:, 0], bar_width)
# plt.bar(index + bar_width, jumento_celestino[:, 1], bar_width)
# plt.axhline(y=np.mean(jumento_celestino[:, 0]), color='b', linestyle='--')
# plt.axhline(y=np.mean(jumento_celestino[:, 1]), color='r', linestyle='--')
# plt.legend(["Mean Test Accuracy", "Mean Validation Accuracy", "Test Accuracy", "Validation Accuracy"])
# plt.show()

print("Mean Test Accuracy: ", np.mean(jumento_celestino[:, 0]))
print("Mean Validation Accuracy: ", np.mean(jumento_celestino[:, 1]))






import pygame
import numpy as np
import random
import time


def send_action(class_):
    # click the corresponding arrow key
    if class_ == 0:
        pydirectinput.press('d')
        print("Right key pressed")
    elif class_ == 1:
        pydirectinput.press('a')
        print("Left key pressed")
    elif class_ == 2:
        pydirectinput.press('w')
        print("Up key pressed")
    elif class_ == 3:
        pydirectinput.press('s')
        print("Down key pressed")
    else:
        print("INVALID CLASS")
   


WINDOW_SIZE = 401
N_WRONG_GUESSES = 0
DETECTION_WINDOW = 600
REST_CLASS = 0
history = [[0, 0, 0, 0]]*(WINDOW_SIZE-50) + [[1, 1, 1, 1]]*50
control_classify = WINDOW_SIZE


X_test = np.array(all_test_data[subject])
X_test = X_test.transpose(1, 2, 0, 3, 4)
# X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2], X_test.shape[3], X_test.shape[4])
print(X_test.shape)


# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Real-Time Event Detection")


print("quee shape", X_test.shape)
QUEUE = X_test[0][0]
input_stream = QUEUE

while QUEUE.shape[2] < 1000:
    chosen_event = random.randint(0, X_test.shape[1] - 1)
    QUEUE = np.concatenate((QUEUE, X_test[0][chosen_event]), axis=2)

#take one sample from the queue and put it in the input stream, then remove it from the queue and remove the first sample from the input stream
def get_sample(QUEUE, input_stream):
    sample = QUEUE[:,:,0]
    sample = np.expand_dims(sample, axis=2)
    QUEUE = QUEUE[:,:,1:]
    input_stream = np.concatenate((input_stream[:, :, 1:], sample), axis=2)
    return QUEUE, input_stream

print(QUEUE.shape, input_stream.shape)

while input_stream.shape[2] < WINDOW_SIZE:
    QUEUE, input_stream = get_sample(QUEUE, input_stream)

running = True
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                class_ = 1
                print("Sent class 1 to QUEUE")
            elif event.key == pygame.K_2:
                class_ = 2
                print("Sent class 2 to QUEUE")
            elif event.key == pygame.K_3:
                class_ = 3
                print("Sent class 3 to QUEUE")
            elif event.key == pygame.K_4:
                class_ = 4
                print("Sent class 4 to QUEUE")
            else:
                class_ = None
            
            if class_:
                # Add event to Queue
                chosen_event = random.randint(0, X_test.shape[1] - 1)
                QUEUE = X_test[class_][chosen_event]


    # Roll the QUEUE and input_stream
    QUEUE, input_stream = get_sample(QUEUE, input_stream)

    # Add random events to the QUEUE if it is too small
    if QUEUE.shape[0] < 400:
        chosen_event = random.randint(0, X_test.shape[1] - 1)       
        QUEUE = np.concatenate((QUEUE, X_test[0][chosen_event]), axis=2)

    # pick the last WINDOW_SIZE samples from the input stream
    sample_data = input_stream[:, :, -WINDOW_SIZE:].reshape(1, input_stream.shape[0], input_stream.shape[1], WINDOW_SIZE)

    # classify the sample
    if control_classify == 0:
        classification = inference(sample_data, chosen_bands, filter_banks, model)[0]
    else:
        control_classify -= 1
        classification = REST_CLASS
    entry = [0, 0, 0, 0]


    # Visual feedback
    screen.fill((0, 0, 0))
    if classification != REST_CLASS:
        entry[int(classification)-1] = 1
        pygame.draw.rect(screen, (255, 0, 0), (0, 0, 800, 600))
        
        #wrtite classification
        font = pygame.font.Font(None, 36)
        text = font.render(str(classification), True, (0, 0, 0))
        screen.blit(text, (200, 150))


    history.append(entry)

    # plot one channel and band of the QUEUE
    channel = 0
    band = 0
    # data = QUEUE[band, channel, :]
    data = sample_data[0, band, channel, :]
    data = data - np.mean(data)
    data = data / np.max(data)
    data = data * 300
    data = data + 300
    for i in range(1, len(data)):
        pygame.draw.line(screen, (0, 255, 0), (i-1, data[i-1]), (i, data[i]), 2)


    # display in a window the history as an image
    history_img = np.array(history)
    if classification != REST_CLASS:
        scores = history_img[-DETECTION_WINDOW:].sum(axis=0)
        if max(scores) > 150:
            print(f"Event {np.argmax(scores)+1} detected")
            send_action(np.argmax(scores))
            history = [[0, 0, 0, 0]]*(WINDOW_SIZE-50) + [[1, 1, 1, 1]]*50
            control_classify = WINDOW_SIZE

        #print(scores)
    history_img = history_img.T*255
    history_img = pygame.surfarray.make_surface(history_img)
    ## resize
    history_img = pygame.transform.scale(history_img, (400, 600))
    screen.blit(history_img, (400, 0))








    pygame.display.flip()
    # time.sleep(0.01)

pygame.quit()
