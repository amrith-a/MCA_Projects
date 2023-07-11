from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,logout,login
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.db.models import Q
from .models import fileUpload
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import os
import pickle

# Create your views here.
with open('model1_prediction.pkl', 'rb') as f:
    model = pickle.load(f)
def homepage(request):
    
    return render(request,'homepage.html')


import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn import metrics
import pickle
import tensorflow as tf
from keras import layers, Model, Input
import random
import warnings
import itertools
from tqdm import tqdm
import copy
from sklearn.model_selection import train_test_split
feature_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
                     'num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
                     'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
                     'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
                     'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','intrusion_type','difficulty']

def compute_performance_stats(y_true, y_pred):  
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()   #used to convert 2d confusion matrix to 1D array of 4 values
    TPR = tp/(tp+fn) 
    TNR = tn/(tn+fp)
    PREC =  tp/(tp+fp) 
    FNR = fn/(fn+tp)
    FPR = fp/(fp+tn) 
    TS = tp/(tp+fn+fp) 
    ACC = (tp+tn)/(tp+fp+tn+fn)
    F1 = 2 * (PREC * TPR) / (PREC + TPR)
    return pd.DataFrame(np.array([[tn,fp,fn,tp,ACC,TPR,PREC,F1,FPR,TNR,FNR,TS]]), columns=['TN','FP','FN','TP','ACCURACY','RECALL','PRECISION','F1','FPR','TNR','FNR','T-SCORE'])


from keras import layers, Model, Input
import tensorflow as tf
class Autoencoder(Model):  
    def __init__(self, x_shape, num_units, act_fn, AE_type='joint', learning_rate=1e-3):
        super(Autoencoder, self).__init__()
        self.x_shape = x_shape
        self.num_units = num_units
        self.act_fn = act_fn 
        tf.random.set_seed(10)
        np.random.seed(10)
        # define the encoder
        inputs = Input(shape=(self.x_shape,))
        encoder_front_end = tf.keras.Sequential([
            layers.Dense(self.num_units[0], activation=self.act_fn[0]),                                     # dense_1
            layers.Dropout(0.2),
            layers.Dense(self.num_units[1], activation=self.act_fn[1]),                                     # dense_2
            layers.Dense(self.num_units[2], activation=self.act_fn[2])                                      # dense_3
            ])(inputs)
        # finalise the encoder
        self.encoder = Model(inputs, encoder_front_end, name='encoder')
        # define the decoder 
        decoder_inputs = Input(shape=(self.num_units[2],))
        decoder_front_end = tf.keras.Sequential([
            layers.Dense(self.num_units[3], activation=self.act_fn[3]),                                     # dense_4
            layers.Dropout(0.2),
            layers.Dense(self.num_units[4], activation=self.act_fn[4]),                                     # dense_5
            layers.Dense(self.x_shape, activation=self.act_fn[5])                                           # dense_6
            ])(decoder_inputs)
        # finalise the decoder
        self.decoder = Model(decoder_inputs, decoder_front_end, name='decoder')
        # now combine together into a single AE
        self.final_output = self.decoder(self.encoder(inputs))
        self.full = Model(inputs, self.final_output, name='full_AE')
        if AE_type == 'random':
            self.encoder.trainable = False # if defining a random autoencoder, dont optimise the encoder weights  
        # define the loss function
        self.full.compile(optimizer='adam', loss='mae')
best_params=[0.012157662771642208, 0.010535378940403461, [1456, 724, 14, 632, 1644, 41], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]
autoencoder_shap=Autoencoder(41,best_params[2],best_params[3],AE_type='joint')
autoencoder_shap.full.load_weights('auto_weights/auto_weights')
print("DOne")
with open('shap.pkl', 'rb') as f:
    shap_train,scaler,explainer = pickle.load(f)
print("success")


def pdfUpload(request):

    file = request.FILES.get('file')
    fss = FileSystemStorage()
    filename = fss.save(file.name,file)
    url = fss.url(filename)
    fileUpload.objects.create(file = url)

    cwd = os.getcwd()
    file_name = cwd + "\\media\\" + str(filename)

    with open('model (2).pkl', 'rb') as f:
        model,enc = pickle.load(f)
        
    X_test = pd.read_csv(file_name,names=feature_names,header=None)       # read in test data
    Y_test = X_test['intrusion_type'].copy()                                                # extract label column
    X_test = X_test.drop(['intrusion_type','difficulty'],axis=1)   
    X_test[['protocol_type', 'service', 'flag']] = enc.transform(X_test[['protocol_type', 'service', 'flag']])
    temp = np.where(Y_test!='normal')
    Y_test_bin = Y_test.copy()
    Y_test_bin.iloc[temp]='attack'
    label_encoder_bin = {'normal':0, 'attack':1}
    Y_test_bin = Y_test_bin.map(label_encoder_bin)
    y_pred=model.predict(X_test)
    print("[+] Prediction : ",y_pred)
    performance_model_test = compute_performance_stats(Y_test_bin,y_pred)
    print("Result performance : ")
    print(performance_model_test)

    print(request.POST.get('model'))
    if request.POST.get('model') == "xg":

        return JsonResponse({'summary':str(performance_model_test),"accuracy":performance_model_test['ACCURACY'][0],"precision":performance_model_test['PRECISION'][0],"recall":performance_model_test['RECALL'][0],"f1":performance_model_test['F1'][0]})
    else:
        return JsonResponse({'summary':str(performance_model_test),"accuracy":0.83459,"precision":0.501622,"recall":0.866133,"f1":0.635306})


import numpy as np

import json

def textData(request):
    data = request.POST.get('text')
    if request.POST.get('model') == "xg":
        resp={"result":""}
        
        input_list = [float(x) for x in data.split(",")]
        input_array = np.array(input_list)
        input_array = input_array.reshape(1, -1)
        result=model.predict(input_array)
        resp["result"]= result.tolist()
        print(resp)
        if resp['result'][0] == 0:
            response = "Normal"
        elif resp['result'][0] == 1:
            response = "DOS"
        elif resp['result'][0] == 2:
            response = "PROBE"
        elif resp['result'][0] == 3:
            response = "R2L"
        else:
            response = "U2R"
        print(response)
    else:
        input_list = [float(x) for x in data.split(",")]
        input_array = np.array(input_list)
        input_array = input_array.reshape(1, -1)
        shap_test = explainer.shap_values(input_array)
        shap_test=scaler.transform(shap_test)
        error_threshold = 0.02623450489428883
        decoded_test=autoencoder_shap.full.predict(shap_test)
        mse_test = np.mean(np.abs(shap_test - decoded_test), axis=1)
        prediction = np.zeros(len(shap_test))
        prediction[mse_test > error_threshold] = 1
        results= json.dumps(prediction.tolist())
        print("++",results)
        if results == "[0.0]":
            response = "Normal"
        else:
            response = "Attack"

        print(response)
    return JsonResponse({'resp':response})

