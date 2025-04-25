import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_random_coordinates(n, a, b):
    random.seed(42)  # Set the seed for reproducibility
    # Calculate the maximum number of possible unique coordinates
    max_unique = (b - a) * (b - a)
    
    if n > max_unique:
        raise ValueError("Cannot create more than " + str(max_unique) + " unique coords")
    
    # Initialize an empty set to store generated coordinates
    unique_coords = set()
    
    while len(unique_coords) < n:
        x = random.randint(a, b)  # Generate a random integer between a and b (inclusive) for x coordinate
        y = random.randint(a, b)  # Generate a random integer between a and b (inclusive) for y coordinate
        
        coord = (x, y)  # Create a tuple of the coordinates
        
        unique_coords.add(coord)  # Add to the set if it's not already present
    
    return list(unique_coords)  # Return the list of generated unique coordinates
def generate_coordinates(n):
    coords = []  # Initialize an empty list to store coordinate tuples
    
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            coord = (x*10,y*10)  # Create a tuple of coordinates
            coords.append(coord)  # Add the tuple to the list of coordinates
    
    return coords  # Return the list of all possible coordinates within the square grid

def create_undirected_graph(coordinates):
    # Create a new empty Graph object
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        x, y = coord  # Extract the coordinates
        
        # Add the current node to the graph with ID as i and set its attributes
        G.add_node(i, x=x, y=y)

    for i in range(len(coordinates)):

        for j in range(i+1, len(coordinates)):
            if(G.has_edge(i, j)):#Si ya existe un edge entre los nodos,
                #se salta la creación de un nuevo edge
                continue
            #Creacion de edges posibles entre nodos
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            #Cálculo del peso del edge (distancia euclidiana)
            peso= ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            #Añadir el edge al grafo
            G.add_edge(i, j, weight=peso)
    return G  # Return the created undirected graph
def calculate_probabilities(dict_edges, alpha, beta):
    for edge in dict_edges:
        dict_edges[edge][2] = (dict_edges[edge][1] )** alpha * (1 / dict_edges[edge][0]) ** beta

def pheromoneAddition(dict_edges, visited, pheromoneRate, evap_rate,best_global_path,best_global_length,noElite):
    #Tao(t+n)=Tao(t)*(1-evap_rate)+pheromoneRate/weight*cantidad_de_hormigas_que_visitaron_el_edge
    # Evaporar feromonas en todos los edges
    #Tao(t)*(1-evap_rate)
    for edge in dict_edges:
        dict_edges[edge][1] *= (1 - evap_rate)
    # Para cada tour de cada hormiga, se añade feromona a los edges visitados en el tour
    #pheromoneRate/weight*cantidad_de_hormigas_que_visitaron_el_edge
    for i in range(len(visited)):
        length = calculate_length(dict_edges, visited[i])
        for j in range(len(visited[i]) - 1):
            #pheromoneRate/weight}
            dict_edges[(visited[i][j], visited[i][j+1])][1] += \
                pheromoneRate / length
            dict_edges[(visited[i][j+1], visited[i][j])][1] += \
                pheromoneRate / length
    #Las hormigas elite refuerzan con mayor cantidad de feromonas los edges que visitaron,
    #en esta implementación estas hormigas no recorren el grafo, solo refuerzan los edges (son espectadoras)
    for j in range(len(best_global_path) - 1):
        dict_edges[(best_global_path[j], best_global_path[j+1])][1] += \
            noElite*pheromoneRate / best_global_length
        dict_edges[(best_global_path[j+1], best_global_path[j])][1] += \
            noElite*pheromoneRate / best_global_length
    
def calculate_length(dict_edges, path):
    return sum([dict_edges[(path[i], path[i+1])][0] for i in range(len(path) - 1)])

def calculate_best_path(dict_edges, paths):
    lengths = [calculate_length(dict_edges, path) for path in paths]
    min_index = np.argmin(lengths)
    return paths[min_index], lengths[min_index]

def caminar(ants_init, G, dict_edges, alpha, beta, evap_rate, pheromoneRate,noNodes,noElite,best_global_path,best_global_length):

    #creacion de un diccionario con los edges y sus pesos para buscarlos más facilmente en menor tiempo
    ants = ants_init.copy()
    visited = [[ant] for ant in ants_init]  # Inicializar la lista de nodos visitados
    visited_set = [set([ant]) for ant in ants_init]
    for _ in range(noNodes - 1):

        for i, ant in enumerate(ants):
            # Obtener vecinos no visitados del nodo actual
            neighbors = [neighbor for neighbor in G.neighbors(ant) if int(neighbor) not in visited_set[i]]

            # Calcular las probabilidades de movimiento
            probs = np.array([dict_edges[(ant, neighbor)][2] for neighbor in neighbors])
            probs /= probs.sum()
            
            # Mover la hormiga al siguiente nodo basado en las probabilidades
            next_node = int(np.random.choice(neighbors, p=probs))
            
            ants[i] = next_node
            visited[i].append(next_node)
            visited_set[i].add(next_node)

    # Añadir el nodo inicial para cerrar el ciclo
    for i in range(len(ants)):
        visited[i].append(ants_init[i])
        visited_set[i].add(ants_init[i])
    best_path, best_length = calculate_best_path(dict_edges, visited)
    if best_length < best_global_length:#Como globlal_length es inicializado como infinito, la primera vez que se ejecute, se actualizará
        best_global_path = best_path
        best_global_length = best_length

    # Actualizar las feromonas de los edges
    pheromoneAddition(dict_edges, visited, pheromoneRate, evap_rate,best_global_path,best_global_length,noElite)
    calculate_probabilities(dict_edges, alpha, beta)
    return best_path, best_length, best_global_path, best_global_length

def ACO(G, evap_rate, noAnts, noElite,noIter, alpha, beta, pheromoneRate, pheroInit):
    dict_edges={}
    for x in G.edges:
        dict_edges[x]=[G[x[0]][x[1]]['weight'],pheroInit,(pheroInit**alpha)*((1/G[x[0]][x[1]]['weight'])**beta)]
        dict_edges[(x[1],x[0])]=[G[x[0]][x[1]]['weight'],pheroInit,(pheroInit**alpha)*((1/G[x[0]][x[1]]['weight'])**beta)]
    ants = [random.randint(0, len(G.nodes) - 1) for _ in range(noAnts)]
    scores = []
    best_global_length=float('inf')
    best_global_path=[]
    for x in range(noIter):
        best_path, best_length, best_global_path, best_global_length = caminar(ants,G, dict_edges, alpha, beta, evap_rate, pheromoneRate,len(G.nodes),noElite,best_global_path,best_global_length)
        scores.append(best_global_length)
        print(f"Iteración {x + 1}: Mejor valor de ruta = {best_length}")
        print(f"Mejor valor de ruta global = {best_global_length}")
        scores.append(best_global_length)
    return scores, (best_global_path, best_global_length)
def plot_path(path, coords, edge_color='black', node_size=200):
    # Create a new empty graph object
    new_graph = nx.Graph()
    
    # Add nodes from the path
    nodes = set(path)
    new_graph.add_nodes_from(nodes, color='blue')

    # Add edges based on consecutive pairs of nodes in the path
    pairs = list(zip(path[:-1], path[1:]))
    edges = [(a, b) for (a, b) in pairs]
    new_graph.add_edges_from(edges, color=edge_color)
    pos={i: (coords[i][0], coords[i][1]) for i in range(len(coords))}
    nx.draw(new_graph, with_labels=True, pos=pos, node_size=node_size, node_color='blue', edge_color=edge_color)
    # Plot the new graph with specified node and edge properties
    plt.show()

#Generación de coordenadas aleatorias con valores de 0 al 100 para sus coordenadas en un espacio de 2D
# coordinates_25 = generate_random_coordinates(25, 0, 100)
# coordinates_25_graph = create_undirected_graph(coordinates_25)
# coordinates_100 = generate_random_coordinates(100, 0, 100)
# coordinates_100_graph = create_undirected_graph(coordinates_100)
coordinates_225 = generate_random_coordinates(225, 0, 100)
coordinates_225_graph = create_undirected_graph(coordinates_225)

#Generación de coordenadas en un espacio de 2D cuadrado de 5x5, 10x10 y 15x15
# coordinates_25_square = generate_coordinates(5)
# coordinates_25_square_graph = create_undirected_graph(coordinates_25_square)
# coordinates_100_square = generate_coordinates(10)
# coordinates_100_square_graph = create_undirected_graph(coordinates_100_square)
coordinates_225_square = generate_coordinates(15)
coordinates_225_square_graph = create_undirected_graph(coordinates_225_square)

# #Optimización de la ruta en el grafo con 25 nodos aleatorios
# solution=ACO(coordinates_25_graph, 0.5, 50, 5,100,2, 5, 50,100)
# print(solution[1])
# plt.figure()
# plt.xscale('log')
# plt.plot(solution[0],label='Optimización de la ruta en el grafo con 25 nodos aleatorios')
# plt.legend()
# plt.figure()
# plot_path(solution[1][0], coordinates_25)
# #Optimización de la ruta en el grafo con 100 nodos aleatorios
# solution=ACO(coordinates_100_graph, 0.6, 100, 10,50,4, 5, 50,100)
# print(solution[1])
# plot_path(solution[1][0], coordinates_100)
#Optimización de la ruta en el grafo con 225 nodos aleatorios

start_time = time.time()
solution=ACO(coordinates_225_graph, 0.3, 75, 7,50, 4, 5, 100,500)
end_time = time.time()

print(solution[1])
print(f"Tiempo de ejecución: {end_time - start_time} segundos")
plt.figure()
plt.xscale('log')
plt.plot(solution[0],label='Optimización de la ruta en el grafo con 225 nodos aleatorios')
plt.legend()
plt.figure()
plot_path(solution[1][0], coordinates_225)



# #Optimización de la ruta en el grafo con 25 nodos en malla de 5x5
# solution=ACO(coordinates_25_square_graph, 0.5, 30, 5,50,2, 5, 50,100)
# print(solution[1])
# plt.figure()
# plt.xscale('log')
# plt.plot(solution[0],label='Optimización de la ruta en el grafo con 25 nodos en malla de 5x5')
# plt.legend()
# plt.figure()
# plot_path(solution[1][0], coordinates_25_square)
# #Optimización de la ruta en el grafo con 100 nodos en malla de 10x10
# solution=ACO(coordinates_100_square_graph, 0.6, 100, 10,50,2, 5, 50,100)
# print(solution[1])

# plot_path(solution[1][0], coordinates_100_square)
# #Optimización de la ruta en el grafo con 225 nodos en malla de 15x15
start_time = time.time()
solution=ACO(coordinates_225_square_graph, 0.3, 75, 7,50, 2, 5, 100,500)
end_time = time.time()

print(solution[1])
print(f"Tiempo de ejecución: {end_time - start_time} segundos")
plt.figure()
plt.xscale('log')
plt.plot(solution[0],label='Optimización de la ruta en el grafo con 225 nodos en malla de 15x15')
plt.legend()
plt.figure()
plot_path(solution[1][0], coordinates_225_square)
