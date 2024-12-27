import pickle
import matplotlib.pyplot as plt

# Carica il file pickle
with open("Plot/Pickle/LMS_error.pkl", "rb") as f:
    fig = pickle.load(f)

# Mostra il grafico interattivo
fig.show()
plt.show()

with open("Plot/Pickle/Task_error.pkl", "rb") as f:
    fig = pickle.load(f)

# Mostra il grafico interattivo
fig.show()
plt.show()

with open("Plot/Pickle/Cup.pkl", "rb") as f:
    fig = pickle.load(f)

# Mostra il grafico interattivo
fig.show()
plt.show()