import os
import dill

def save_object(obj, file_path):
    dir_path = os.path.dirname(file_path)

    # Creating Directory
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file_obj:
        dill.dump(obj,file_obj)
    
def load_object(file_path):

    with open(file_path,'rb') as file_obj:
        return dill.load(file_obj)
