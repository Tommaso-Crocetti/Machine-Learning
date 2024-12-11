import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_neural_network(network, input, output):
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
    
    for layer, n_units in enumerate(layers):
        y_offset = -(n_units/2)
        if layer != len(layers) - 1:
            node_id_minus_one = f"L{layer}_N{-1}"
            G.add_node(node_id_minus_one)  
            node_sizes [node_id_minus_one] = 500
            node_labels[node_id_minus_one] = "1.00"
            node_colors[node_id_minus_one] = "yellow"
            pos[node_id_minus_one] = (layer / 2, -np.max(layers)-2)
        
        for unit in range(n_units):
            node_id = f"L{layer}_N{unit}"
            G.add_node(node_id)
            if layer == 0:
                y_offset = -(2*(n_units-1))/2
                pos[node_id] = (layer / 2, y_offset + 2*unit)
                node_sizes [node_id] = 500
                node_labels[node_id] = f"{input[unit]:.2f}"  
                node_colors[node_id] = "gray"
            elif layer < len(layers)-1: 
                y_offset = -(8*(n_units-1))/2
                pos[node_id] = (layer / 2, y_offset + 8*unit)
                node_sizes [node_id] = 1500
                node_labels[node_id] = f"{network.store_hidden_result[layer-1][unit]:.2f}"  
                node_colors[node_id] = "green"
            else:
                y_offset = -(n_units/2)
                pos[node_id] = (layer / 2, y_offset + 4*unit)
                node_sizes [node_id] = 1500
                node_labels[node_id] = f"{output[0]:.2f}"
                node_colors[node_id] = "green"

    #mette gli edge tra i neuroni
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
    
    #disegna la figura
    node_colors_list = np.array(list(node_colors.values()))
    node_sizes_list = np.array(list(node_sizes.values()))
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=node_sizes_list)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=5, arrows=False)

    plt.show()

def weight_to_color (x):
    return [0, 0, 1] if x < 0 else [1, 0, 0]

#scelgo la trasparenza in base al valore assoluto del rapporto tra il peso considerato e il valore assoluto del peso massimo
def weight_alpha(x): 
    alpha = np.abs(np.divide(x,1))
    return alpha
        
