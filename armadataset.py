import os
import cv2
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

os.system ("cls")
ruta_originales = 'originales'
ruta_editadas = 'editadas'
ruta_guardado = ''

data_list = [] #aca voy a guardar la info que saque de las fotos
i=0
#este bloque captura todos los archivos para saber el total
archivos = []
for root, _, files in os.walk(ruta_originales):
  for filename in files:
    if filename.endswith(('.jpg', '.png')):
      #ruta de las imagenes espejo, una de editada, otra de original
      inicio = time.time()
      ruta_imagen_original = os.path.join(root, filename)
      ruta_imagen_editada = os.path.join(ruta_editadas, os.path.relpath(ruta_imagen_original, ruta_originales))
      archivos.append({"o":ruta_imagen_original,"e":ruta_imagen_editada})
    
with tqdm(total=len(archivos)) as pbar:    
  for a in archivos:
    #capturo la imagen con opencv. Vienen en BGR
    pbar.update(1)
    imagen_original = cv2.imread(a["o"])
    imagen_editada = cv2.imread(a["e"])
    pbar.set_description(f"Procesando {a['o'][10:]}")
    
    proceder = False
    

    try:
      if imagen_original == None:
        proceder = False
        #print(f"salteando {ruta_imagen_original}. El archivo no pudo ser abierto")
    except ValueError:
      proceder = True
      try:
        if imagen_editada == None:
          #print(f"Salteando {ruta_imagen_editada}. El archivo no pudo ser abierto")
          proceder = False
      except ValueError:
        proceder = True
  
    if not proceder:
      continue
    
    if imagen_original.shape != imagen_editada.shape:
      #print("Imagenes de tamaño diferente. Se saltean")
      continue
    #hago un promedio de los canales individuales para simplificar un poco el problema
    imagen_promedio_original = np.floor(np.mean(imagen_original, axis=2)).astype(np.uint8)
    imagen_promedio_editada = np.floor(np.mean(imagen_editada, axis=2)).astype(np.uint8)

    #consulto el pixel mas negro en la imagen editada y me fijo si no era negro antes y encuentro la diferencia entre los dos para encontrar el valor aproximado de negro
    #lo mismo con los blancos. Consulto el 255 en editada que no es 255 en original y saco la diferencia aproximada. No funciona simepre pero da una aproximacion muy buena
    diferencia = np.where((imagen_promedio_editada == 255) & (imagen_promedio_original != 255), imagen_promedio_editada - imagen_promedio_original, 0).astype(np.int16)
    diferencia2 = np.where((imagen_promedio_editada == 0) & (imagen_promedio_original != 0), imagen_promedio_original, 0).astype(np.int16)

    blanco = 255 - np.max(diferencia) #le resto 255 a la diferencia para obtener el valor de blanco a aplicar
    negro = np.max(diferencia2)

    #hay casos en donde el valor de negro no es exactamente negro o era negro y luego se modifico un poco algun canal y toma valores horribles
    #en ese caso se toma un valor aproximado promedio entre 11 y 13 para negro y de 217 a 219 para blanco
    if negro < 10 and negro < 199:
      negro = random.randint(11,13)
      blanco= random.randint(217,219)

    factor_escala = 0.05  # 10% del tamaño original

    # Calcula las nuevas dimensiones de la imagen
    nuevas_dimensiones = (int(imagen_original.shape[1] * factor_escala), int(imagen_original.shape[0] * factor_escala))

    # Redimensiona la imagen
    imagen_original_c = cv2.resize(imagen_original, nuevas_dimensiones, interpolation=cv2.INTER_LINEAR)
    imagen_editada_c = cv2.resize(imagen_editada, nuevas_dimensiones, interpolation=cv2.INTER_LINEAR)

    #error alto para minimizar
    mse_min = 200.0

    gamma = 120 #el limite de gamma lo pongo en 1,2 porque es demasiado procesamiento por imagen sino
    s=gamma
    t=gamma
    u=gamma
    gamma_correcto = [1.0,1.0,1.0]
    
    pbar.set_description(f"Analizando estilo {a['o'][10:]}")
    for s in range(90, gamma, 1): #comienzo en 0,9 y subo de a 0,1 por vez
      for t in range(90, gamma, 1):
        for u in range(90, gamma, 1):
          inBlack = np.array([negro,negro,negro], dtype=np.float32)
          inWhite = np.array([blanco,blanco,blanco], dtype=np.float32)
          inGamma = np.array([s/100,t/100,u/100], dtype=np.float32)
          outBlack= np.array([0,0,0], dtype=np.float32)
          outWhite = np.array([255,255,255], dtype=np.float32)
          #la maldita funcion gamma
          img = np.clip( (imagen_original_c - inBlack) / (inWhite - inBlack), 0, 255)
          img = ( img ** (1/inGamma)) * (outWhite - outBlack) + outBlack
          img = np.clip( img, 0, 255).astype(np.uint8)

          mse = np.mean((img - imagen_editada_c)**2)
          if mse < mse_min and mse != 1.0:
            #print(f"Mejorando mse: antes {mse_min}, ahora {mse}. gamma: {[s/100,t/100,u/100]}")
            mse_min = mse
            gamma_correcto = [s/100,t/100,u/100]
    #control de posibles errores que puedan arruinar el flujo del programa
    if imagen_original is not None:
        imagen_o_hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV)
    else:
        #print(f"Error: No se pudo cargar la imagen original. {filename}")
        continue

    if imagen_editada is not None:
        imagen_e_hsv = cv2.cvtColor(imagen_editada, cv2.COLOR_BGR2HSV)
    else:
        #print(f"Error: No se pudo cargar la imagen editada. {filename}")
        continue

    #a partir de aca trato de sacar toda la informacion estadistica posible para guardar la menor cantidad de informacion posible pero siendo
    #al mismo tiempo lo mas representativo que se puede... Creo que es un buen enfoque para identificar futuras fotos. Obtengo HSV y RGB
    matiz = imagen_o_hsv[:, :, 0]
    saturacion = imagen_o_hsv[:, :, 1]
    valor = imagen_o_hsv[:, :, 2]

    promedio_o_matiz = np.mean(matiz)
    mediana_o_matiz = np.median(matiz)
    std_o_matiz = np.std(matiz)

    promedio_o_saturacion = np.mean(saturacion)
    mediana_o_saturacion = np.median(saturacion)
    std_o_saturacion = np.std(saturacion)

    promedio_o_valor = np.mean(valor)
    mediana_o_valor = np.median(valor)
    std_o_valor = np.std(valor)

    resumen_hsv_H_o = promedio_o_matiz + mediana_o_matiz + std_o_matiz
    resumen_hsv_S_o = promedio_o_saturacion + mediana_o_saturacion + std_o_saturacion
    resumen_hsv_V_o = promedio_o_valor + mediana_o_valor + std_o_valor


    canal_rojo = imagen_original[:, :, 0]
    canal_verde = imagen_original[:, :, 1]
    canal_azul = imagen_original[:, :, 2]

    media_rojo = np.mean(canal_rojo)
    media_verde = np.mean(canal_verde)
    media_azul = np.mean(canal_azul)

    mediana_rojo = np.median(canal_rojo)
    mediana_verde = np.median(canal_verde)
    mediana_azul = np.median(canal_azul)

    desviacion_estandar_rojo = np.std(canal_rojo)
    desviacion_estandar_verde = np.std(canal_verde)
    desviacion_estandar_azul = np.std(canal_azul)

    resumen_histograma = (
        media_rojo + media_verde + media_azul +
        mediana_rojo + mediana_verde + mediana_azul +
        desviacion_estandar_rojo + desviacion_estandar_verde + desviacion_estandar_azul
    )
    fin = time.time()
    
    #if (mse_min > 50):
    #  print("\n",filename, "IMAGEN CON MUCHO MARGEN DE ERROR!")
    #print("\n",ruta_imagen_original,"MSE:",mse_min, "blancos:",blanco,"negros:",negro,"gamma (BGR):",gamma_correcto)
    #print("Procesada en",np.round(fin-inicio,2),"segundos.",f"Tiempo estimado para {3000-i} fotos:",np.round(((fin-inicio)*(3000-i))/60,2), "minutos")
    i+=1
    pbar.set_description(f"Agregando al dataset {a['o'][10:]}")
    data_list.append({"NAME":ruta_imagen_original,"HUE_AVG": promedio_o_matiz, "HUE_MDN": mediana_o_matiz, "HUE_STD": std_o_matiz,
                      "SATURATION_AVG": promedio_o_saturacion, "SATURATION_MDN": mediana_o_saturacion, "SATURATION_STD": std_o_saturacion,
                      "VALUE_AVG": promedio_o_valor, "VALUE_MDN": mediana_o_valor, "VALUE_STD": std_o_valor,
                      "RED_AVG":media_rojo,"RED_MDN":mediana_rojo,"RED_STD":desviacion_estandar_rojo,
                      "GREEN_AVG":media_verde,"GREEN_MDN":mediana_verde,"GREEN_STD":desviacion_estandar_verde,
                      "BLUE_AVG":media_azul,"BLUE_MDN":mediana_azul,"BLUE_STD":desviacion_estandar_azul,
                      "HIST": resumen_histograma, "WHITES": blanco, "BLACKS": negro, "GAMMA_R":gamma_correcto[2],
                      "GAMMA_G":gamma_correcto[1], "GAMMA_B":gamma_correcto[0], "MSE":mse_min})
    #if i % 10 == 0:
    #  print(f"Procesadas hasta el momento: {i} imagenes")
    #i +=1
  #if i > 500:
    # break
  #if i > 500:
   # break
dataset = "axis300.csv"
df = pd.DataFrame(data_list)
df.to_csv(ruta_guardado+dataset, index=False)
print(f"Analisis finalizado. Dataset creado como {dataset}.\nProcesadas {i} imagenes de un total de {len(archivos)}")