import os
import zipfile
import gdown

'''
Download and extract NER dataset
'''


def download_dataset():
    URL_train = "https://drive.google.com/uc?export=download&id=1-pX9s6EjSxj9_gIRQ8ll_ScnQcbaCSNW"
    URL_dev = "https://drive.google.com/uc?export=download&id=1-prAPYiFsACjLTQvSzwKt5oRV1l1pSba"
    URL_test = "https://drive.google.com/uc?export=download&id=1-qKhBwagoN7UroNk6sFEi4gVQbgT_VOn"
    URL_glove = "https://drive.google.com/uc?export=download&id=1nIwgYeAoALTAZGqiVGpAczPO8JY2CTlr"
    
    gdown.download(URL_glove, os.getcwd()+os.sep+"glove.6B.50d.txt", quiet=False)
    gdown.download(URL_train, os.getcwd()+os.sep+"train.json", quiet=False)
    gdown.download(URL_dev, os.getcwd()+os.sep+"dev.json", quiet=False)
    gdown.download(URL_test, os.getcwd()+os.sep+"test.json", quiet=False)
    print("\n")


if __name__ == '__main__':

    download_dataset()