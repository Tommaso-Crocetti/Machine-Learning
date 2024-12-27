import pickle
import matplotlib.pyplot as plt

def show_plot(id):
    with open("Plot/Pickle/" + str(id) + "LMS_error.pkl", "rb") as f:
        fig = pickle.load(f)

    # Mostra il grafico interattivo
    fig.show()
    plt.show()

    with open("Plot/Pickle/" + str(id) + "Task_error.pkl", "rb") as f:
        fig = pickle.load(f)

    # Mostra il grafico interattivo
    fig.show()
    plt.show()

    with open("Plot/Pickle/Cup.pkl", "rb") as f:
        fig = pickle.load(f)

    # Mostra il grafico interattivo
    fig.show()
    plt.show()