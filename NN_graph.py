import matplotlib.pyplot as plt
import networkx as nx


def plot_neural_network(layers, weights=None, outputs=None):
    G = nx.DiGraph()  # Creiamo un grafo orientato per rappresentare la rete

    # Creazione dei nodi e posizionamento
    pos = {}
    node_labels = {}
    node_colors = {}
    y_offset = 0
    
    for layer, n_units in enumerate(layers):
        y_offset = -(n_units / 2)
        
        #mette i neuroni
        if layer != len(layers) - 1:
            node_id_minus_one = f"L{layer}_N{-1}"
            G.add_node(node_id_minus_one)  
            node_labels[node_id_minus_one] = "1.00"
            node_colors[node_id_minus_one] = "yellow"
            pos[node_id_minus_one] = (layer / 2, y_offset - 1)
        
        for unit in range(n_units):
            node_id = f"L{layer}_N{unit}"
            G.add_node(node_id)
            pos[node_id] = (layer / 2, y_offset + unit)
            if outputs:
                node_labels[node_id] = f"{outputs.get(node_id, 0):.2f}"  
                node_colors[node_id] = "green"

    node_colors_list = [node_colors.get(node, "black") for node in G.nodes]

    #mette gli edge tra i neuroni
    for layer in range(len(layers) - 1):
        for src in range(-1, layers[layer]):
            for dest in range(layers[layer + 1]):
                src_id = f"L{layer}_N{src}"
                dest_id = f"L{layer + 1}_N{dest}"
                G.add_edge(src_id, dest_id)
                if weights:
                    G[src_id][dest_id]["weight"] = weights.get((src_id, dest_id), 0.0)  

    #dovrebbe colorare ma non lo fa
    edge_colors = []
    if weights:
        for u, v in G.edges():
            weight = G[u][v]["weight"] 
            #color = weight_to_color(weight)
            #alpha = weight_alpha(weight)
            #edge_colors.append((color[0], color[1], color[2], alpha)) 
    else:
        edge_colors = ["gray"] * len(G.edges()) 

    #disegna la figura
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1500, node_color=node_colors_list)
    edge_width = 10 
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=edge_width, arrows=False) #al posto di "gray" dovrebbe esserci edge_colors, ma non funziona a modo

    plt.show()


layers = [3,4,2] 
weights = {
    ("L0_N0", "L1_N0"): 0.6,
    ("L0_N1", "L1_N1"): -0.4,
    ("L0_N1", "L1_N2"): 100,  
} 
outputs = {"L0_N0": 1.0, "L1_N0": 0.8}  #uscite dei neuroni

plot_neural_network(layers, weights, outputs)
