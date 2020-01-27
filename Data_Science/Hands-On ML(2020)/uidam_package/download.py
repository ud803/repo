import urllib
import os
import gzip
import re

def unzipTar(file_name, file_path, extract_path):
    print("Please add unzipTar Method")
    return 1

def unzipZip(file_name, file_path, extract_path):
    print("Please add unzipZip Method")
    return 1

def unzipGzip(file_name, file_path):
    extract_path = file_path[:-3]
    with gzip.open(file_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print("...file unzipped : {}".format(extract_path))
    return extract_path

    
def downloadAndUnzipFile(dataset_name, data_dict, overwrite=False):
    if not os.path.isdir(os.path.join("datasets", dataset_name)):
        os.makedirs(os.path.join("datasets", dataset_name))
    
    url_splitter = re.compile('/')
    
    print("Download Start...")
    for item in data_dict.keys():
        url = data_dict[item]
        file_name = url_splitter.split(url)[-1]
        file_path = os.path.join("datasets", dataset_name, file_name)
        file_url = data_dict[item]
        if(overwrite==False and os.path.exists(file_path)):
            print("...File Already Exists. Use overwrite=True to overwrite.")
            break
        res = urllib.request.urlretrieve(file_url, file_path)
        print("...file downloaded : {}".format(file_name))

    print("Download Completed...")

    print("Unzip Start...")
    for item in data_dict.keys():
        url = data_dict[item]
        file_name = url_splitter.split(url)[-1]
        file_path = os.path.join("datasets", dataset_name, file_name)
        if(overwrite==False and os.path.exists(file_path)):
            print("...File Already Exists. Use overwrite=True to overwrite.")
            break
        file_name = url_splitter.split(data_dict[item])[-1]
        while(re.compile('.*\.(gz|zip|tar)$').search(file_name)):
            file_path = os.path.join("datasets", dataset_name, file_name)
            if(re.compile('.*\.gz$').search(file_name)):
                file_name = unzipGzip(file_name, file_path)
    print("Unzip Completed...")