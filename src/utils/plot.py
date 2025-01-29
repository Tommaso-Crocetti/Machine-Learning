import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_error(errors, other_errors = None, other_label = None, folder = None, filename = None, save = True, show = False):
    plot = plt.figure(figsize=(8, 6))
    plt.plot(range(len(errors)), errors, c='blue', label='Train')
    if other_errors:
        plt.plot(range(len(other_errors)), other_errors, c='red', linestyle='--', label=other_label)
    plt.title(filename)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    if folder and filename:
        if save:
            plt.savefig("Plot/" + folder + "/" + filename + ".png")
    elif filename:
        if save:
            plt.savefig("Plot/" + filename + ".png")
    if show:
        plt.show()
    plt.close()

def plot_output(network, X, y = None, filename = "", save = False, show = False):
    data = []
    for i in range(len(X)):
        output = pd.Series(network.network_output(X.iloc[i]))
        data.append({
            'target_x': output.iloc[0],
            'target_y': output.iloc[1],
            'target_z': output.iloc[2],
        })
    data = pd.DataFrame(data)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data['target_x'], data['target_y'], data['target_z'], c='red', marker='o', label="Network output", s=50)  
    if y is not None:
        ax.scatter(y['target_x'], y['target_y'], y['target_z'], c='blue', marker='o', label="Target value", s=50)  

    plt.legend()


    ax.set_title(filename)
    ax.set_xlabel('Asse X')
    ax.set_ylabel('Asse Y')
    ax.set_zlabel('Asse Z')

    x_min = min(y['target_x'].min(), data['target_x'].min()) if y is not None else data['target_x'].min()
    x_max = max(y['target_x'].max(), data['target_x'].max()) if y is not None else data['target_x'].max()
    y_min = min(y['target_y'].min(), data['target_y'].min()) if y is not None else data['target_y'].min()
    y_max = max(y['target_y'].max(), data['target_y'].max()) if y is not None else data['target_y'].max()
    z_min = min(y['target_z'].min(), data['target_z'].min()) if y is not None else data['target_z'].min()
    z_max = max(y['target_z'].max(), data['target_z'].max()) if y is not None else data['target_z'].max()

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    plt.tight_layout()

    if save:
        plt.savefig("Plot/" + filename + ".png")
    if show:
        plt.show()
    plt.close()

def plot(network, filename):
    
    def weight_to_color (x):
        return [0, 0, 1] if x < 0 else [1, 0, 0]


    def weight_alpha(x): 
        alpha = np.abs(np.divide(x,2))
        return alpha

    G = nx.DiGraph()
    layers = []
    layers.append(network.input_layer.neurons)
    for i in range(network.depth):
        layers.append(network.hidden_layers[i].neurons)
    layers.append(network.output_layer.neurons)

    pos = {}
    node_labels = {}
    node_colors = {}
    node_sizes = {}
    y_offset = 0
    max_layers = np.max(layers[1:])

    for layer, n_units in enumerate(layers):
        if layer != len(layers) - 1:
            y_offset = -(n_units/2)
            node_id_minus_one = f"L{layer}_N{-1}"
            G.add_node(node_id_minus_one)  
            node_sizes [node_id_minus_one] = 200
            node_labels[node_id_minus_one] = " "
            node_colors[node_id_minus_one] = "yellow"
            pos[node_id_minus_one] = (layer / 2, - (max_layers/2+1))

        for unit in range(n_units):
            node_id = f"L{layer}_N{unit}"
            G.add_node(node_id)
            if layer == 0:
                y_offset = -((n_units-1))/2
                pos[node_id] = (layer / 2, y_offset + unit)
                node_sizes [node_id] = 200
                node_labels[node_id] = " "  
                node_colors[node_id] = "gray"
            elif layer < len(layers)-1: 
                y_offset = -((n_units-1))/2
                pos[node_id] = (layer / 2, y_offset + unit)
                node_sizes [node_id] = 200
                node_labels[node_id] = " "  
                node_colors[node_id] = "green"
            else:
                y_offset = -((n_units-1))/2
                pos[node_id] = (layer / 2, y_offset + unit)
                node_sizes [node_id] = 200
                node_labels[node_id] = " "
                node_colors[node_id] = "green"


    for layer in range(network.depth+1):
        for src in range(-1, layers[layer]):
            for dest in range(layers[layer + 1]):
                src_id = f"L{layer}_N{src}"
                dest_id = f"L{layer + 1}_N{dest}"
                if (layer < network.depth):
                    G.add_edge(src_id, dest_id)
                    G[src_id][dest_id]["weight"] = network.hidden_layers[layer].weight_matrix[dest][src]
                elif (layer == ((network.depth))):
                    G.add_edge(src_id, dest_id)
                    G[src_id][dest_id]["weight"] = network.output_layer.weight_matrix[dest][src]


    edge_colors = []
    for u, v in G.edges():
        weight = G[u][v]["weight"] 
        color = weight_to_color(weight)
        alpha = weight_alpha(weight)
        edge_colors.append((color[0], color[1], color[2], alpha)) 


    node_colors_list = np.array(list(node_colors.values()))
    node_sizes_list = np.array(list(node_sizes.values()))
    fig = plt.figure(figsize=(16, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=node_sizes_list, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=3, arrows=False)

    plt.axis('off')
    plt.title("Neural Network")
    plt.tight_layout()

    plt.show()