import os
from GeneradorDatos import carpeta

def borrar_archivo():
    ls = os.listdir()
    for archivo in ls:
        if archivo[-5] != 'K':
            os.remove(archivo)

borrar_archivo()