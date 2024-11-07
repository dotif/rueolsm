
#  Sistema de Reconocimiento y Ubicación Espacial de Objetos en un Laboratorio de Síntesis de Materiales

En este trabajo se desarrollo un sistema de visión por computadora para un laboratorio de síntesis de materiales. Se utiliza YOLO-NAS para el reconocimiento de objetos y una técnica de triangulación para la estimación de profundidad.


## Descripción de Módulos

### Aumento de datos
El objetivo de este modulo es realizar el aumento de datos esta compuesto por 3 archivos:

#### _dataAugmentor.config_

Este archivo centraliza la personalización del proceso. En él, se especifican:

- Rutas: Directorios de origen de las imágenes y etiquetas, así como las ubicaciones de destino para las imágenes procesadas, nuevas etiquetas y visualizaciones con bounding boxes.
* Nomenclatura: Un prefijo o sufijo que se añadirá a los nombres de archivo originales para facilitar la identificación.
+ Formatos: Las extensiones de imagen válidas para el procesamiento.
- Clases: Las categorías de objetos que el modelo está entrenado para detectar.

#### _DataAugmentor.py_
#### _main.py_
### Calibracion de camara
### Entrenamiento
### Predicción



## Autor

- [@dotif](https://github.com/dotif)
