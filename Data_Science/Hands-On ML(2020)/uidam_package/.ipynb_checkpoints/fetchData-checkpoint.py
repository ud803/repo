import os
import struct as st
import numpy as np

def fetch_mnist():
    path = os.path.join("datasets", "mnist")
    
    file_dict = {
        'X_train' : "train-images-idx3-ubyte",
        'y_train' : "train-labels-idx1-ubyte",
        'X_test' : "t10k-images-idx3-ubyte",
        'y_test' : "t10k-labels-idx1-ubyte"
    }
    
    file_list = []
    
    for file in file_dict.keys():
        file_path = os.path.join(path, file_dict[file])
        with open(file_path, 'rb') as f_in:
            if('X' in file):
                magic = st.unpack('>4B', f_in.read(4))
                nImg = st.unpack('>I', f_in.read(4))[0]
                nRow = st.unpack('>I', f_in.read(4))[0]
                nCol = st.unpack('>I', f_in.read(4))[0]
                nBytesTotal = nImg*nRow*nCol*1
                images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal, f_in.read(nBytesTotal))).reshape((nImg,nRow,nCol))
                file_list.append(images_array)
            else:
                magic = st.unpack('>4B', f_in.read(4))
                nImg = st.unpack('>I', f_in.read(4))[0]
                label_array = np.asarray(st.unpack('>'+'B'*nImg, f_in.read(nImg)))
                file_list.append(label_array)
    return file_list