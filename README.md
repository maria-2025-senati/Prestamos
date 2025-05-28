# Predicción del Monto de Préstamos Digitales con Machine Learning y Redes Neuronales

## Objetivo del trabajo
El objetivo principal de este proyecto es predecir el promedio del saldo en préstamos digitales (3 meses) en base al comportamiento transaccional digital de los clientes. Para ello, se emplean técnicas de regresión lineal y redes neuronales artificiales, con el fin de explorar la relación entre variables y mejorar la capacidad predictiva.

## Descripción del Dataset
El dataset utilizado se denomina `dataBasePrestDigital.csv` y contiene información transaccional de clientes. Las columnas principales incluyen:

- `promTrxDig3Um`: Promedio de transacciones digitales en los últimos 3 meses.
- `promSaldoPrest3Um`: Promedio del saldo en préstamos digitales.
- `promSaldoBanco3Um`: Promedio del saldo en cuentas bancarias.
- `genero`, `rngEdad`, `rngSueldo`: Características demográficas del cliente.
- Otras variables relevantes relacionadas con campañas, tarjetas, y uso de canales digitales.

## Librerías utilizadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Explicación de los Modelos
-----------------------------

### 📌 Regresión Lineal Simple

Se utilizó la variable `promTrxDig3Um` para predecir `promSaldoPrest3Um`.

Se entrenó el modelo con `train_test_split` y se graficaron los valores reales vs predichos.

### 📌 Red Neuronal con Keras

Variables usadas:

- `promTrxDig3Um`
- `promSaldoBanco3Um`
- `frecCamp`

Modelo:

- 1 capa oculta con activación `relu`.
- Capa de salida con activación `linear`.
- Pérdida: `mse`, métrica: `mae`.
- Épocas: 100.

📉 Resultados y Gráficas
------------------------

- Regresión Lineal.
- Red Neuronal - Error vs Epochs.
- Predicción de valores.

🧠 Lógica de Programación Implementada
--------------------------------------

Se clasificaron los clientes en niveles de transacciones (bajo, medio, alto).

Se utilizó un diccionario, listas, bucle `for` y condicionales para contar y visualizar la distribución.

📝 Conclusiones Personales
--------------------------

- Existe una relación directa entre la actividad digital de los clientes y su promedio de saldo en préstamos.
- La red neuronal mejora la capacidad predictiva al considerar múltiples variables.
- Aprendí a aplicar técnicas de ML y Deep Learning en un caso real con datos de clientes.
- GitHub es una herramienta clave para documentar y compartir proyectos de análisis de datos.
