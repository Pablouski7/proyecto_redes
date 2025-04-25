# Clase para implementar el ACO AS
# Implementa la lógica del algoritmo ACO AS para resolver el problema del TSP
# tanto para grados aleatorios como para grafos nxn. 

from .ant import Ant
import math
import random
import copy
import gc
from matplotlib import pyplot as plt
import networkx as nx
import threading
import matplotlib

matplotlib.use('Agg')

file_lock = threading.Lock()

PATH = "src/results"

class ACO_AS:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, Q, nxn = True):
        self.max_epochs = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        if nxn:
            if num_ants**0.5 != int(num_ants**0.5):
                raise ValueError("num_ants must be a perfect square for nxn graphs")
            n_nodos_lado = int(num_ants**0.5)
            self.posiciones =  generar_grafo_nxn(n_nodos_lado)
        else:
            self.posiciones = generar_grafo_aleatorio_en_espacio(num_ants)
        self.edges = generar_edges_fully_connected(self.posiciones)
        ants = []
        for i in range(num_ants):
            ant = Ant(i, alpha, beta)
            ants.append(ant)
        self.ants = ants
        del ants

        self.nxn = nxn

    def actualizar_intensidad_rastro(self):
        for edge in self.edges:
            self.edges[edge] = (self.rho * self.edges[edge][0] + self.calcular_delta_tau(edge), self.edges[edge][1])
    
    def calcular_delta_tau(self, edge):
        delta_tau = 0.0
        for ant in self.ants:
            edges_path = list(zip(ant.tabu, ant.tabu[1:]))
            if edge in edges_path:
                delta_tau += self.Q / ant.tour_length
        return delta_tau
    
    def run(self):
        best_ant_global = Ant(None, self.alpha, self.beta)
        scores = []
        stagnation = False
        for epoch in range(self.max_epochs):
            if stagnation:
                break
            for i in range(len(self.posiciones)):
                for ant in self.ants:
                    ant.move(self.posiciones, self.edges)
            best_ant_epoch = min(self.ants)
            scores.append(best_ant_epoch.tour_length)
            
            cnt_iguales = 0
            for ant in self.ants:
                if ant == best_ant_global:
                    cnt_iguales += 1
            if cnt_iguales == len(self.ants):
                stagnation = True
                print(f"Stagnation reached at epoch {epoch+1}")
                break

            # print(f"Best Ant (Epoch {epoch+1}): {best_ant_epoch.tabu} - Tour Length: {best_ant_epoch.tour_length}")

            if best_ant_epoch < best_ant_global:
                best_ant_global = copy.deepcopy(best_ant_epoch)
            
            self.actualizar_intensidad_rastro()
            for ant in self.ants:
                ant.tour_length = 0.0
                ant.tabu = [ant.tabu[0]]
                ant.current_node = ant.tabu[0]

        if self.nxn:
            n = int(len(self.posiciones)**0.5)
            print(f"Results for {n}x{n} cities using ACO_AS with hyperparameters: epochs:{self.max_epochs}, alpha={self.alpha}, beta={self.beta}, rho={self.rho}, Q={self.Q}")
            print(f"Best Ant: {best_ant_global.tabu} - Tour Length: {best_ant_global.tour_length}")
            plot_graph(self.posiciones, best_ant_global.tabu, best_ant_global.tabu[0])
            plot_scores(scores, "Best Global Fitness through Epochs", self.posiciones, self.max_epochs, None)
        else:
            print(f"Results for {len(self.posiciones)} cities using ACO_AS with hyperparameters: epochs:{self.max_epochs}, alpha={self.alpha}, beta={self.beta}, rho={self.rho}, Q={self.Q}")
            print(f"Best Ant: {best_ant_global.tabu} - Tour Length: {best_ant_global.tour_length}")
            plot_graph(self.posiciones, best_ant_global.tabu, best_ant_global.tabu[0], nxn=False)
            plot_scores(scores, "Best Global Fitness through Epochs", self.posiciones, self.max_epochs, None, nxn=False)
        
        gc.collect()
        return best_ant_global
    
def generar_edges_fully_connected(posiciones):
    c = 1
    nodos = list(posiciones.keys())
    num_nodos = len(nodos)
    if num_nodos <= 0:
        return []  # No hay nodos para conectar
    
    edges = set()  # Utilizamos un conjunto para evitar duplicados automáticamente
    for i in range(num_nodos):
        for j in range(i + 1, num_nodos):  # Comenzamos desde i+1 para evitar aristas duplicadas
            # Almacenamos la arista en orden ascendente para consistencia (evita (2,1) si ya hay (1,2))
            edge = tuple(sorted((nodos[i], nodos[j])))
            edges.add(edge)
    
    edges_dict = {}
    for edge in edges:
        x1, y1 = posiciones[edge[0]]
        x2, y2 = posiciones[edge[1]]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        edges_dict[edge] = (c, dist)

    return edges_dict

def generar_grafo_aleatorio_en_espacio(N, ancho_espacio=100, alto_espacio=100, seed=7):
    if seed is not None:
        random.seed(seed)
    
    generados = set()
    posiciones = {}
    for nodo in range(N):
        while True:
            x = random.randint(0, ancho_espacio - 1)
            y = random.randint(0, alto_espacio - 1)
            if (x, y) not in generados:
                generados.add((x, y))
                posiciones[nodo] = (x, y)
                break

    if seed is not None:
        random.seed()

    return posiciones

def generar_grafo_nxn(nodos_por_lado, separacion = 10, seed=8):
    if seed is not None:
        random.seed(seed)

    posiciones = {}
    for i in range(nodos_por_lado):
        for j in range(nodos_por_lado):
            nodo = i * nodos_por_lado + j
            posiciones[nodo] = (j * separacion, i * separacion)
    
    if seed is not None:
        random.seed()

    return posiciones

def plot_scores(scores, title, posiciones, max_epochs=None, stop_epoch=None, nxn=True):
    with file_lock:  # Asegura que solo un hilo pueda ejecutar este bloque a la vez
        # Normaliza la longitud de scores si max_epochs está definido
        if max_epochs is not None:
            last_score = scores[-1] if scores else 0  # Obtiene el último score o usa 0 si la lista está vacía
            scores_normalized = scores + [last_score] * (max_epochs - len(scores))
        else:
            scores_normalized = scores

        # Plotea los scores normalizados
        plt.plot(range(len(scores_normalized)), scores_normalized, label="Global bests through epochs", color='r')

        # Agrega anotación para la época de parada si corresponde
        if stop_epoch is not None:
            plt.axvline(x=stop_epoch, color='g', linestyle='--', label=f'Stop Epoch: {stop_epoch}')
            plt.legend()  # Asegúrate de mostrar la leyenda para la línea de stop epoch

        plt.title(title)
        plt.ylabel("Best Global Fitness")
        plt.xlabel("Epochs")

        if nxn:
            filename = f"nxn_{len(posiciones)}_epochs_plot.png"
        else:   
            filename = f"space_{len(posiciones)}_epochs_plot.png"

        plt.savefig(f"{PATH}/{filename}")
        plt.close()

# Función para graficar el grafo de ciudades con la mejor solución encontrada y guardar el plot
def plot_graph(posiciones, best_chromosome, ciudad_inicio, nxn=True):
    with file_lock:  # Asegura que solo un hilo pueda ejecutar este bloque a la vez
        edges = gen_edges_de_cromosoma(best_chromosome)
        G = nx.Graph()
        G.add_nodes_from(posiciones.keys())
        G.add_edges_from(edges)
        color_map = []
        for node in G:
            if node == ciudad_inicio:
                color_map.append('green')
            else:
                color_map.append('skyblue')
        nx.draw(G, pos=posiciones, node_color=color_map, with_labels=False, node_size=20)
        plt.axis('equal')

        if nxn:
            plt.savefig(f"{PATH}/nxn_{len(posiciones)}_tsp_solution_plot.png")
        else:
            plt.savefig(f"{PATH}/space_{len(posiciones)}_tsp_solution_plot.png")
        plt.close()

# Función para generar las aristas de un cromosoma para TSP hecha por nemotron:latest
def gen_edges_de_cromosoma(cromosoma):
    edges = list(zip(cromosoma, cromosoma[1:])) # Básicamente empareja cada ciudad con la siguiente excepto la última
    return edges