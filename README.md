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
