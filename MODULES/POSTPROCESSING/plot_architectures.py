#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:32:17 2023

@author: afernandez
"""
import tensorflow as tf
from visualkeras import layered_view
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
input_dim=(10,38)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_dim),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same", name='C1'),
    layers.BatchNormalization(),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Plot the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

# Save the model figure as a PDF
with PdfPages('model_architecture.pdf') as pdf:
    fig = plt.figure(figsize=(8, 6), dpi=500)
    plt.imshow(plt.imread('model.png'))
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()



plt.figure()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
# model.summary()
plt.savefig('model_architecture.pdf', bbox_inches='tight', dpi=150)

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

input_dim = (10, 38)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_dim),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same", name='C1'),
    layers.BatchNormalization(),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling1D(),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate the plot without layer names
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

# Load the saved image
image = plt.imread('model.png')

# Create a new figure and subplot
fig, ax = plt.subplots(figsize=(10, 5))

# # Display the first half of the architecture
# ax.imshow(image, extent=[0, 0.5, 0, 1])
# ax.axis('off')

# Add a connecting line
arrowprops = dict(facecolor='black', arrowstyle='|-|', lw=1.5)
ax.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.5), arrowprops=arrowprops)

# Display the second half of the architecture with offset
ab = AnnotationBbox(OffsetImage(image, zoom=0.5), (0.5, 0.5), frameon=False, xybox=(0.5, 0.5))
ax.add_artist(ab)

# Save the custom layout as an image
plt.savefig('split_model_architecture.pdf', bbox_inches='tight', dpi=150)

# Display the custom layout (optional)
plt.show()



