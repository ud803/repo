import os
import joblib
import re


def file_exists(file_path):
    if os.path.exists(file_path):
        return True
    
def model_exists(model_name):
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        return True
    else:
        return False

def get_model(model_name):
    path = os.path.join("models", model_name)
    if(file_exists(path)):
        print("model loaded...")
        return joblib.load(path)
    else:
        print("model does not exists.")
        return None

def save_model(model, model_name):
    re_pkl = re.compile('\.pkl$')
    if(re_pkl.search(model_name) == None):
        print("wrong model Name")
        return
    
    path = os.path.join("models", model_name)
    if(file_exists(path)):
        print("model already exists.")
        return
    else:
        print("model saved")
        joblib.dump(model, path)
    