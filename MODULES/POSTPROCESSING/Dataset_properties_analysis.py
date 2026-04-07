#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:13:04 2025

@author: afernandez
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
# --- 1. Datos extraídos de la tabla ---
# Columnas: Valores de L_sim
l_sim_values = [20, 50, 80, 100, 300]
# Filas: Valores de n_t
n_t_values = [5, 10, 30, 50, 100]

# Matriz de Accuracy. Usamos np.nan para el valor faltante ("-").
accuracy_data = np.array([
    [0.28, 0.59, 0.64, 0.69, 0.66],
    [0.28, 0.61, 0.71, 0.70, 0.66],
    [0.51, 0.77, 0.86, 0.86, 0.84],
    [0.70, 0.90, 0.93, 0.95, 0.93],
    [np.nan, 1.00, 0.99, 1.00, 0.98]
])

# --- 2. Configuración del estilo del gráfico ---
# Usamos un estilo más adecuado para presentación con curvas
plt.style.use('seaborn-v0_8-talk')
fig, ax = plt.subplots(figsize=(14, 9))

# MODIFICACIÓN 2: Poner un fondo gris claro a la figura
ax.set_facecolor('#e0e0e0') # Color de fondo del área del gráfico
fig.patch.set_facecolor('white') # Color de fondo fuera del gráfico

# --- 3. Creación de las curvas ---
# Transponemos los datos para que cada fila sea una curva de L_sim
accuracy_by_lsim = accuracy_data.T

# Paleta de colores para las líneas
colors = plt.cm.viridis(np.linspace(0, 1, len(l_sim_values)))

for i, l_sim in enumerate(l_sim_values):
    acc_values = accuracy_by_lsim[i]
    
    # Resaltamos la curva de L_sim = 100, que es la de interés
    if l_sim == 100:
        linewidth = 5.0
        linestyle = '-'
        marker = 'o'
        markersize = 12
        label = f'$L_{{sim}} = {l_sim}$ s'
        zorder = 10 # Para que se dibuje por encima de las demás
    else:
        linewidth = 5.0
        linestyle = '--'
        marker = 'x'
        markersize = 7
        label = f'$L_{{sim}} = {l_sim}$ s'
        zorder = 5

    ax.plot(n_t_values, acc_values, marker=marker, linestyle=linestyle,
            linewidth=linewidth, color=colors[i], label=label, zorder=zorder,
            markersize=markersize)

# --- 4. Resaltar y anotar el compromiso (trade-off) ---
# Coordenadas de los puntos de interés en la curva de L_sim = 100
lsim100_acc = accuracy_by_lsim[l_sim_values.index(100)]
nt50_acc = lsim100_acc[n_t_values.index(50)]


# --- 5. Títulos, etiquetas y leyenda ---
ax.set_xlabel('$n_t$ (delay indicator)', fontsize=26)
ax.set_ylabel('Test Accuracy', fontsize=26)

# MODIFICACIÓN 1: Poner fontsize 20 para los ticks de los ejes
ax.tick_params(axis='both', which='major', labelsize=26)

# Resaltamos el punto n_t=50 en la curva óptima con un círculo
ax.plot(50, nt50_acc, 'o', markersize=22, fillstyle='none', color='red', mew=5.5, label=' Selected solution')

# MODIFICACIÓN 3: Poner fondo opaco a la leyenda
legend = ax.legend(fontsize=26, loc='lower right')
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8) # Ajusta la opacidad aquí

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')

# Mejorar los límites de los ejes para mayor claridad
ax.set_ylim(0.2, 1.05)
ax.set_xlim(0, 110)

# --- 6. Mostrar el gráfico ---
plt.tight_layout()
plt.savefig(os.path.join("Output", "A_final_outputs", 'dataproperties_comparison_analysis.png'), dpi=500, bbox_inches='tight')
plt.show()

