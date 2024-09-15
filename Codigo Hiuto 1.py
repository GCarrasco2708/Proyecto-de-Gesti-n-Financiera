# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:54:41 2024

@author: gcarr
"""


import numpy as np
import scipy.stats as si
from datetime import datetime

# Función de Black-Scholes para una opción Call
def black_scholes_call(S, K, T, r, sigma):
    # Cálculo de d1 y d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Cálculo del precio de la opción
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

# Parámetros de la opción
S0 = 120.90  # Precio del activo subyacente
K = 117      # Precio de ejercicio
r = 0.0366   # Tasa libre de riesgo
sigma = 0.26  # Volatilidad implícita (ajustar si es necesario)

# Tiempo hasta la expiración (en años fraccionarios)
fecha_actual = datetime.now()
fecha_vencimiento = datetime(2024, 9, 27)  # Fecha de vencimiento de la opción
T = (fecha_vencimiento - fecha_actual).days / 365  # Tiempo en años fraccionarios

# Cálculo del precio de la opción utilizando Black-Scholes
call_price_bs = black_scholes_call(S0, K, T, r, sigma)
print(f"El precio de la opción con el modelo de Black-Scholes es: {call_price_bs:.2f}")