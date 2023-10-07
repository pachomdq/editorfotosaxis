import os
import pandas as pd
import joblib
import cv2
import numpy as np
from PIL import Image

ruta_originales = 'originales'
ruta_editadas = 'nuevas'
print("cargando modelo...",end="")
modelo_cargado = joblib.load('modelofotosaxis.pkl')
print("ok!")

def crear_estructura_directorios(origen, destino):
    for root, _, _ in os.walk(origen):
        estructura_relativa = os.path.relpath(root, origen)
        directorio_destino = os.path.join(destino, estructura_relativa)
        if not os.path.exists(directorio_destino):
            os.makedirs(directorio_destino)

crear_estructura_directorios(ruta_originales, ruta_editadas)

#esto queda obsoleto
if not (os.path.exists(ruta_editadas)):
    try:
        os.makedirs(ruta_editadas)    
    except OSError as e:
        print(f'Error al crear la ruta: {e}')

for root, _, files in os.walk(ruta_originales):
  for filename in files:
    if filename.endswith(('.jpg', '.png', '.JPG', '.PNG')):
        ruta_imagen_original = os.path.join(root, filename)
        ruta_imagen_editada = os.path.join(ruta_editadas, os.path.relpath(ruta_imagen_original, ruta_originales))
        if not (os.path.exists(ruta_editadas)):
            try:
                os.makedirs(ruta_editadas)    
            except OSError as e:
                print(f'Error al crear la ruta: {e}')
        print(f"Analizando {filename}...")
        imagen_original = cv2.imread(ruta_imagen_original)

        if imagen_original is not None:
            imagen_o_hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV)
        else:
            print(f"Error: No se pudo cargar la imagen original.")
            continue

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


        #'HUE_AVG', 'HUE_MDN', 'HUE_STD', 'SATURATION_AVG',
        #'SATURATION_MDN', 'SATURATION_STD', 'VALUE_AVG', 'VALUE_MDN', 'VALUE_STD', 'HIST'
        # Hacer predicciones en nuevos datos utilizando el modelo cargado
        nuevos_datos = pd.DataFrame({'HUE_AVG': [promedio_o_matiz],
                                    'HUE_MDN': [mediana_o_matiz],
                                    'HUE_STD': [std_o_matiz],
                                    'SATURATION_AVG':[promedio_o_saturacion],
                                    'SATURATION_MDN':[mediana_o_saturacion],
                                    'SATURATION_STD':[std_o_saturacion],
                                    'VALUE_AVG':[promedio_o_valor],
                                    'VALUE_MDN':[mediana_o_valor],
                                    'VALUE_STD':[std_o_valor],
                                    'RED_AVG':[media_rojo],
                                    'RED_MDN':[mediana_rojo],
                                    'RED_STD':[desviacion_estandar_rojo],
                                    'GREEN_AVG':[media_verde],
                                    'GREEN_MDN':[mediana_verde],
                                    'GREEN_STD':[desviacion_estandar_verde],
                                    'BLUE_AVG':[media_azul],
                                    'BLUE_MDN':[mediana_azul],
                                    'BLUE_STD':[desviacion_estandar_azul],
                                    'HIST':[resumen_histograma]
                                    })
        prediccion = modelo_cargado.predict(nuevos_datos)
        w = np.round(prediccion[0,0]).astype(int)
        b = np.round(prediccion[0,1]).astype(int)
        gr = np.round(prediccion[0,2],2)
        gg = np.round(prediccion[0,3],2)
        gb = np.round(prediccion[0,4],2)
        print("Generando parametros de edicion:", "blanco",w, "negro", b, "gammaR", gr, "gammaG", gg, "gammaB", gb)
        inBlack = np.array([b,b,b], dtype=np.float32)
        inWhite = np.array([w,w,w], dtype=np.float32)
        #BGR
        inGamma = np.array([gb,gg,gr], dtype=np.float32)
        outBlack= np.array([0,0,0], dtype=np.float32)
        outWhite = np.array([255,255,255], dtype=np.float32)

        img = np.clip( (imagen_original - inBlack) / (inWhite - inBlack), 0, 255)
        img = ( img ** (1/inGamma)) * (outWhite - outBlack) + outBlack
        img = np.clip( img, 0, 255).astype(np.uint8)
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #aca modifico el dpi y lo guardo con pillow porque opencv no tiene la capacidad de modificar el dpi
        dpi = (400, 400)
        img_pil.save(ruta_imagen_editada, dpi=dpi, quality=100)
        #imagen_opencv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Guardar la imagen OpenCV en disco porque el hijodemilputa pil me arruina la calidad aunque le ponga sin compresion
        #cv2.imwrite(ruta_imagen_editada,imagen_opencv,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print("Archivo editado con exito!")
