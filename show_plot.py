import pickle
import matplotlib.pyplot as plt

# Carica il file pickle
with open("Plot/Pickle/training_error.pkl", "rb") as f:
    fig = pickle.load(f)

# Mostra il grafico interattivo
fig.show()
plt.show()

with open("Plot/Pickle/validation_error.pkl", "rb") as f:
    fig = pickle.load(f)

# Mostra il grafico interattivo
fig.show()
plt.show()