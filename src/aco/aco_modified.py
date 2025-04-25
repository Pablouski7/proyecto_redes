# -------- MANTENER LA CLASE Ant COMO ESTABA --------
import random

class Ant:
    def __init__(self, starting_node, alpha, beta):
        if starting_node is None:
            self.tabu = []
            self.tour_length = float("inf")
            self.alpha = alpha
            self.beta = beta
        else:
            self.tabu = [starting_node]
            self.current_node = starting_node
            self.tour_length = 0.0
            self.alpha = alpha  # Parámetro para controlar la importancia del rastro
            self.beta = beta  # Parámetro para controlar la importancia de la visibilidad

    def __str__(self):
        if self.tour_length == float("inf"):
            return "[]|inf"
        # Asegurarse que tabu y tour_length sean strings antes de concatenar
        return str(self.tabu) + '|' + str(self.tour_length)

    def __lt__(self, other):
        return self.tour_length < other.tour_length

    def __eq__(self, other):
        # Compara si dos hormigas han realizado exactamente el mismo tour
        return self.tabu == other.tabu and self.tour_length == other.tour_length

    def move(self, posciones, edges):
        # Si ya visitó todos los nodos, regresa al inicio
        if len(posciones) == len(self.tabu):
            # Asegurar que la arista exista antes de accederla
            edge_key = tuple(sorted((self.current_node, self.tabu[0])))
            if edge_key in edges:
                _, dist = edges[edge_key]
                self.tour_length += dist
                self.tabu.append(self.tabu[0])
                self.current_node = self.tabu[0] # Actualizar nodo actual aunque no se use más en move
            else:
                # Manejar caso donde la arista de cierre no existe (no debería pasar en fully_connected)
                print(f"Error: Arista de cierre {edge_key} no encontrada.")
                # Podrías asignar una penalidad grande o manejar de otra forma
                self.tour_length = float('inf')
            return

        # Nodos que aún no ha visitado
        no_visitados = set(posciones.keys()) - set(self.tabu)
        if not no_visitados: # Si no quedan nodos por visitar (caso raro, pero por seguridad)
             # Intentar cerrar el ciclo como arriba
            edge_key = tuple(sorted((self.current_node, self.tabu[0])))
            if edge_key in edges:
                _, dist = edges[edge_key]
                self.tour_length += dist
                self.tabu.append(self.tabu[0])
                self.current_node = self.tabu[0]
            else:
                self.tour_length = float('inf')
            return

        probabilidades = {}
        denominador = 0.0 # Usar float para los cálculos

        # Calcular el denominador de la fórmula de probabilidad
        for nodo in no_visitados:
            edge_key = tuple(sorted((self.current_node, nodo)))
            if edge_key in edges:
                tau_ik, dist_ik = edges[edge_key]
                # Evitar división por cero si la distancia es 0 (nodos superpuestos)
                eta_ik = (1.0 / dist_ik) if dist_ik > 0 else float('inf')
                # Manejar eta_ik infinito (distancia 0), darle máxima prioridad si alpha es 0
                # o manejarlo de otra forma si alpha > 0 y tau es 0.
                # Una opción simple es asignarle un valor muy grande si dist_ik es 0.
                if dist_ik == 0: eta_ik = 1e18 # Un valor muy grande
                
                prob_component = (tau_ik ** self.alpha) * (eta_ik ** self.beta)
                # Evitar NaN si tau=0 y alpha=0 (0^0 es indefinido, usualmente se toma como 1)
                if tau_ik == 0 and self.alpha == 0: prob_component = (eta_ik ** self.beta)
                
                denominador += prob_component
            else:
                # La arista no existe, no debería pasar en fully connected
                 print(f"Error: Arista {edge_key} no encontrada durante cálculo de denominador.")


        # Calcular probabilidades para cada nodo no visitado
        if denominador == 0: # Si todas las opciones tienen probabilidad 0 (raro)
            # Elegir uno al azar o manejar el error
            if no_visitados:
                choosen_next_nodo = random.choice(list(no_visitados))
            else: # No hay a donde moverse
                 # Intentar cerrar el ciclo
                edge_key = tuple(sorted((self.current_node, self.tabu[0])))
                if edge_key in edges:
                    _, dist = edges[edge_key]
                    self.tour_length += dist
                    self.tabu.append(self.tabu[0])
                    self.current_node = self.tabu[0]
                else:
                    self.tour_length = float('inf')
                return

        else:
            for next_nodo in no_visitados:
                edge_key = tuple(sorted((self.current_node, next_nodo)))
                if edge_key in edges:
                    tau_ij, dist_ij = edges[edge_key]
                    eta_ij = (1.0 / dist_ij) if dist_ij > 0 else float('inf')
                    if dist_ij == 0: eta_ij = 1e18

                    numerador = (tau_ij ** self.alpha) * (eta_ij ** self.beta)
                    if tau_ij == 0 and self.alpha == 0: numerador = (eta_ij ** self.beta)
                    
                    probabilidades[next_nodo] = numerador / denominador
                else:
                    # Arista no existe
                     print(f"Error: Arista {edge_key} no encontrada durante cálculo de probabilidad.")
                     probabilidades[next_nodo] = 0.0 # Asignar probabilidad 0

            # Filtrar nodos con probabilidad calculada (evita errores si alguna arista faltó)
            valid_nodes = list(probabilidades.keys())
            valid_probs = list(probabilidades.values())
            
            # Normalizar probabilidades por si acaso no suman 1 debido a errores de punto flotante
            prob_sum = sum(valid_probs)
            if prob_sum > 0:
                normalized_probs = [p / prob_sum for p in valid_probs]
            else: # Si todas las probabilidades son 0, elegir al azar entre los válidos
                 if valid_nodes:
                     normalized_probs = [1.0/len(valid_nodes)] * len(valid_nodes)
                 else: # No hay nodos válidos a donde moverse
                     # Intentar cerrar el ciclo
                    edge_key = tuple(sorted((self.current_node, self.tabu[0])))
                    if edge_key in edges:
                        _, dist = edges[edge_key]
                        self.tour_length += dist
                        self.tabu.append(self.tabu[0])
                        self.current_node = self.tabu[0]
                    else:
                        self.tour_length = float('inf')
                    return


            # Elegir el siguiente nodo basado en las probabilidades normalizadas
            if not valid_nodes: # Si por alguna razon no hay nodos válidos
                 # Intentar cerrar el ciclo
                edge_key = tuple(sorted((self.current_node, self.tabu[0])))
                if edge_key in edges:
                    _, dist = edges[edge_key]
                    self.tour_length += dist
                    self.tabu.append(self.tabu[0])
                    self.current_node = self.tabu[0]
                else:
                    self.tour_length = float('inf')
                return
                
            choosen_next_nodo = random.choices(valid_nodes, weights=normalized_probs, k=1)[0]

        # Mover la hormiga al nodo elegido
        edge_key = tuple(sorted((self.current_node, choosen_next_nodo)))
        if edge_key in edges:
             _, dist = edges[edge_key]
             self.tour_length += dist
             self.tabu.append(choosen_next_nodo)
             self.current_node = choosen_next_nodo
        else:
             # Error crítico si la arista elegida no existe
             print(f"Error CRÍTICO: Arista elegida {edge_key} no encontrada.")
             self.tour_length = float('inf') # Penalizar fuertemente


# -------- CLASE ACO_AS MODIFICADA --------
from ant import Ant # Asegúrate que la clase Ant esté definida arriba o importada
import math
import random
import copy
import gc
from matplotlib import pyplot as plt
import networkx as nx
import threading
import matplotlib

# Funciones auxiliares (generar_grafo_nxn, generar_grafo_aleatorio, etc.)
# ASUMIMOS QUE ESTÁN DEFINIDAS AQUÍ O IMPORTADAS
# Por ejemplo:
def generar_grafo_nxn(nodos_por_lado, separacion = 10, seed=8):
    # ... (código como en el original) ...
    posiciones = {}
    if seed is not None: random.seed(seed)
    for i in range(nodos_por_lado):
        for j in range(nodos_por_lado):
            nodo = i * nodos_por_lado + j
            posiciones[nodo] = (j * separacion, i * separacion)
    if seed is not None: random.seed() # Reset seed
    return posiciones

def generar_grafo_aleatorio_en_espacio(N, ancho_espacio=100, alto_espacio=100, seed=7):
     # ... (código como en el original) ...
    posiciones = {}
    if seed is not None: random.seed(seed)
    generados = set()
    for nodo in range(N):
        while True:
            x = random.randint(0, ancho_espacio - 1)
            y = random.randint(0, alto_espacio - 1)
            if (x, y) not in generados:
                generados.add((x, y))
                posiciones[nodo] = (x, y)
                break
    if seed is not None: random.seed() # Reset seed
    return posiciones


def generar_edges_fully_connected(posiciones, initial_pheromone=1.0):
    # ... (código como en el original, pero permitiendo initial_pheromone) ...
    nodos = list(posiciones.keys())
    num_nodos = len(nodos)
    if num_nodos <= 0: return {}
    edges_dict = {}
    for i in range(num_nodos):
        for j in range(i + 1, num_nodos):
            edge = tuple(sorted((nodos[i], nodos[j])))
            x1, y1 = posiciones[edge[0]]
            x2, y2 = posiciones[edge[1]]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Asegurar que la distancia no sea cero para evitar división por cero más adelante
            if dist == 0:
                dist = 1e-9 # Asignar una distancia muy pequeña pero no cero
            edges_dict[edge] = (initial_pheromone, dist) # (feromona, distancia)
    return edges_dict

# Funciones de ploteo (plot_graph, plot_scores, gen_edges_de_cromosoma)
# ASUMIMOS QUE ESTÁN DEFINIDAS AQUÍ O IMPORTADAS
def gen_edges_de_cromosoma(cromosoma):
    # ... (código como en el original) ...
    # Asegurarse que el cromosoma tiene al menos 2 elementos para formar una arista
    if len(cromosoma) < 2: return []
    # Genera tuplas (nodo_i, nodo_{i+1}) asegurando el orden para coincidir con las claves de 'edges'
    edges = [tuple(sorted((cromosoma[i], cromosoma[i+1]))) for i in range(len(cromosoma) - 1)]
    # No incluir arista de cierre aquí si las claves de edges no la incluyen
    return edges


matplotlib.use('Agg')
file_lock = threading.Lock()
PATH = "results_eas" # Cambiar path si se desea

# Crear directorio si no existe
import os
import time
if not os.path.exists(PATH):
    os.makedirs(PATH)

class ACO_EAS: # Renombrado para reflejar el cambio a EAS
    # Añadido elite_weight, initial_pheromone y pheromone_min/max
    def __init__(self, num_nodes, num_ants, num_iterations, alpha, beta, rho, Q,
                 elite_weight=1.0, initial_pheromone=1.0, pheromone_min=0.01, pheromone_max=10.0,
                 nxn=True, seed=None):
        self.max_epochs = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho # Tasa de evaporación
        self.Q = Q
        self.elite_weight = elite_weight # Peso para la mejor hormiga global (EAS)
        self.pheromone_min = pheromone_min # Límite inferior de feromona (MMAS opcional)
        self.pheromone_max = pheromone_max # Límite superior de feromona (MMAS opcional)


        self.num_nodes = num_nodes
        if nxn:
             if num_nodes**0.5 != int(num_nodes**0.5):
                 raise ValueError("num_nodes must be a perfect square for nxn graphs")
             n_nodos_lado = int(num_nodes**0.5)
             self.posiciones = generar_grafo_nxn(n_nodos_lado, seed=seed)
        else:
             # Asegurar que num_ants no sea usado para generar el grafo aleatorio
             self.posiciones = generar_grafo_aleatorio_en_espacio(num_nodes, seed=seed)

        # Usar initial_pheromone al generar las aristas
        self.edges = generar_edges_fully_connected(self.posiciones, initial_pheromone)

        # Crear hormigas - el número de hormigas es ahora un parámetro separado
        self.num_ants = num_ants
        ants = []
        # Asignar nodos iniciales distintos si es posible, o aleatorios
        start_nodes = random.sample(list(self.posiciones.keys()), min(num_ants, num_nodes))
        for i in range(num_ants):
             # Si hay más hormigas que nodos, se repetirán nodos iniciales
            start_node = start_nodes[i % len(start_nodes)]
            ant = Ant(start_node, alpha, beta)
            ants.append(ant)
        self.ants = ants
        del ants # Liberar memoria

        self.nxn = nxn


    def actualizar_intensidad_rastro_eas(self, best_ant_global):
        """Actualiza la feromona usando la regla Elitist Ant System (EAS)."""

        # 1. Evaporación en todas las aristas
        for edge in self.edges:
            old_pheromone, dist = self.edges[edge]
            # Aplicar evaporación: tau = (1 - rho) * tau
            new_pheromone_base = (1 - self.rho) * old_pheromone
            self.edges[edge] = (new_pheromone_base, dist)

        # 2. Depósito de feromona por todas las hormigas (AS)
        delta_tau_all = {} # Almacena la suma de deltas para cada arista
        for ant in self.ants:
            # Evitar depósito si el tour es inválido (longitud infinita)
            if ant.tour_length == float("inf") or ant.tour_length == 0: continue

            # Obtener las aristas del tour de la hormiga (asegurando orden)
            ant_edges = gen_edges_de_cromosoma(ant.tabu)
            deposit_amount = self.Q / ant.tour_length

            for edge in ant_edges:
                 # Sumar el depósito a la arista correspondiente
                if edge in self.edges: # Verificar que la arista existe
                    delta_tau_all[edge] = delta_tau_all.get(edge, 0.0) + deposit_amount

        # Sumar los depósitos acumulados a la feromona base evaporada
        for edge, delta in delta_tau_all.items():
            pheromone_base, dist = self.edges[edge]
            self.edges[edge] = (pheromone_base + delta, dist)


        # 3. Depósito Adicional Elitista (EAS)
        if best_ant_global.tour_length != float("inf") and best_ant_global.tour_length != 0:
            # Obtener las aristas del mejor camino global (asegurando orden)
            best_global_edges = gen_edges_de_cromosoma(best_ant_global.tabu)
            # Calcular el depósito elitista extra
            elite_deposit = self.elite_weight * self.Q / best_ant_global.tour_length

            for edge in best_global_edges:
                 if edge in self.edges: # Verificar que la arista existe
                    pheromone_current, dist = self.edges[edge]
                    # Añadir el depósito elitista
                    self.edges[edge] = (pheromone_current + elite_deposit, dist)

        # 4. Opcional: Aplicar límites de feromona (Min-Max Ant System - MMAS)
        # Descomentar si se quiere usar MMAS junto con EAS
        # for edge in self.edges:
        #    pheromone, dist = self.edges[edge]
        #    pheromone = max(self.pheromone_min, min(pheromone, self.pheromone_max))
        #    self.edges[edge] = (pheromone, dist)


    # Esta función ya no es necesaria si calculamos el delta total en actualizar_intensidad_rastro_eas
    # def calcular_delta_tau(self, edge):
    #     ...

    def run(self):
        best_ant_global = Ant(None, self.alpha, self.beta) # Hormiga global vacía inicialmente
        scores = [] # Almacenar la longitud del mejor tour global en cada época
        stagnation_counter = 0
        stagnation_limit = 50 # Número de épocas sin mejora para considerar estancamiento (ajustable)
        last_best_score = float('inf')

        print(f"Iniciando ACO-EAS: {self.num_nodes} nodos, {self.num_ants} hormigas, {self.max_epochs} épocas")
        print(f"Parámetros: alpha={self.alpha}, beta={self.beta}, rho={self.rho}, Q={self.Q}, elite_weight={self.elite_weight}")

        for epoch in range(self.max_epochs):
            # Construir tours para todas las hormigas
            current_epoch_ants = [] # Para almacenar las hormigas con sus tours completos
            for ant in self.ants:
                 # Mover la hormiga hasta completar el tour (visitar todos los nodos + volver al inicio)
                 for _ in range(self.num_nodes): # Mover N veces (N-1 para visitar todos + 1 para cerrar)
                     ant.move(self.posiciones, self.edges)
                     # Si el tour se completa antes (ya tiene N+1 nodos), romper
                     if len(ant.tabu) == self.num_nodes + 1:
                         break
                 # Asegurar que el tour sea válido (cerrado y longitud finita)
                 if len(ant.tabu) != self.num_nodes + 1 or ant.tour_length == float('inf'):
                      # Marcar como inválida o darle longitud infinita si no se completó bien
                      ant.tour_length = float('inf')
                 current_epoch_ants.append(copy.deepcopy(ant)) # Guardar estado final de la hormiga

            # Encontrar la mejor hormiga de *esta* época
            if not current_epoch_ants: # Si ninguna hormiga completó el tour
                print(f"Época {epoch+1}: Ninguna hormiga completó un tour válido.")
                best_ant_epoch = Ant(None, self.alpha, self.beta) # Crear una hormiga inválida
            else:
                # Filtrar hormigas inválidas antes de buscar el mínimo
                valid_ants = [a for a in current_epoch_ants if a.tour_length != float('inf')]
                if not valid_ants:
                     print(f"Época {epoch+1}: Ninguna hormiga completó un tour válido.")
                     best_ant_epoch = Ant(None, self.alpha, self.beta)
                else:
                    best_ant_epoch = min(valid_ants) # Encontrar la mejor entre las válidas

            # Actualizar la mejor hormiga global si la de esta época es mejor
            if best_ant_epoch < best_ant_global:
                best_ant_global = copy.deepcopy(best_ant_epoch)
                print(f"Época {epoch+1}: Nuevo mejor global encontrado! Longitud: {best_ant_global.tour_length:.2f}")
                stagnation_counter = 0 # Resetear contador de estancamiento
            else:
                stagnation_counter += 1

            scores.append(best_ant_global.tour_length) # Guardar el mejor score global hasta ahora
            if (epoch + 1) % 10 == 0: # Imprimir progreso cada 10 épocas
                epoch_best_len = best_ant_epoch.tour_length
                epoch_best_str = f"{epoch_best_len:.2f}" if epoch_best_len != float('inf') else 'inf'
                print(f"Época {epoch+1}/{self.max_epochs} - Mejor Global: {best_ant_global.tour_length:.2f} "
                    f"(Mejor Época: {epoch_best_str})")


            # Comprobar estancamiento basado en falta de mejora
            if stagnation_counter >= stagnation_limit:
                 print(f"Estancamiento detectado en época {epoch+1}: Sin mejora en {stagnation_limit} épocas.")
                 break # Detener si no hay mejora

            # Actualizar feromonas usando la regla EAS y pasando la mejor global
            self.actualizar_intensidad_rastro_eas(best_ant_global)

            # Resetear hormigas para la siguiente época (mantener nodo inicial)
            for ant in self.ants:
                start_node = ant.tabu[0] # Guardar nodo inicial
                ant.__init__(start_node, self.alpha, self.beta) # Reinicializar estado

        # --- Fin del bucle de épocas ---

        print("-" * 30)
        graph_type = f"{int(self.num_nodes**0.5)}x{int(self.num_nodes**0.5)}" if self.nxn else f"{self.num_nodes}_random"
        if best_ant_global.tour_length == float('inf'):
             print(f"No se encontró una solución válida para {graph_type}.")
        else:
            print(f"Resultados finales para {graph_type} usando ACO-EAS:")
            print(f"Parámetros: epochs={self.max_epochs}, ants={self.num_ants}, alpha={self.alpha}, beta={self.beta}, rho={self.rho}, Q={self.Q}, elite_weight={self.elite_weight}")
            print(f"Mejor Longitud de Tour Global: {best_ant_global.tour_length:.4f}")
            print(f"Mejor Camino Global: {best_ant_global.tabu}") # Descomentar si se quiere ver el camino

            # Graficar resultados (asegúrate que las funciones plot_ estén definidas)
            plot_graph(self.posiciones, best_ant_global.tabu, best_ant_global.tabu[0], nxn=self.nxn)
            plot_scores(scores, f"Convergencia ACO-EAS ({graph_type})", self.posiciones, self.max_epochs, epoch+1 if stagnation_counter >= stagnation_limit else None, nxn=self.nxn)
            print(f"Gráficos guardados en la carpeta '{PATH}' (si las funciones de ploteo están activas).")

        gc.collect() # Limpiar memoria
        return best_ant_global, scores

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

# --- Ejemplo de cómo usar la clase ACO_EAS ---
if __name__ == "__main__":
    # Configuración del problema y parámetros del algoritmo
    problema_nxn = True       # True para cuadrícula, False para aleatorio
    num_nodos = 225            # Número total de nodos (si nxn=True, debe ser un cuadrado perfecto)
    
    num_hormigas = 75         # Número de hormigas
    num_epocas = 50          # Número de iteraciones/épocas
    alfa = 4.0                # Importancia de la feromona
    beta = 5.0                # Importancia de la visibilidad (distancia)
    rho = 0.3                 # Tasa de evaporación de feromona (e.g., 0.1 = 10% evaporación)
    Q_val = 100.0             # Cantidad de feromona (factor Q)
    
    peso_elitista = 7.0       # Peso extra para la mejor hormiga global (EAS)
                              # Un valor de num_hormigas es común, o un valor fijo.
    feromona_inicial = 500.0    # Valor inicial de feromona en las aristas
    
    semilla_aleatoria = 42    # Para reproducibilidad

    # Crear instancia del algoritmo
    aco_eas_solver = ACO_EAS(num_nodes=num_nodos,
                             num_ants=num_hormigas,
                             num_iterations=num_epocas,
                             alpha=alfa,
                             beta=beta,
                             rho=rho,
                             Q=Q_val,
                             elite_weight=peso_elitista,
                             initial_pheromone=feromona_inicial,
                             nxn=problema_nxn,
                             seed=semilla_aleatoria)

    # Ejecutar el algoritmo

    start_time = time.time()
    mejor_hormiga, historial_scores = aco_eas_solver.run()
    end_time = time.time()
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

    # Aquí podrías añadir código para usar/visualizar mejor_hormiga y historial_scores
    # Por ejemplo, si las funciones de ploteo están definidas:
    # try:
    #    if mejor_hormiga.tour_length != float('inf'):
    #        plot_graph(aco_eas_solver.posiciones, mejor_hormiga.tabu, mejor_hormiga.tabu[0], nxn=aco_eas_solver.nxn)
    #        plot_scores(historial_scores, f"Convergencia ACO-EAS ({'nxn' if problema_nxn else 'random'} {num_nodos})",
    #                    aco_eas_solver.posiciones, num_epocas, len(historial_scores), nxn=aco_eas_solver.nxn)
    # except NameError:
    #    print("\nAdvertencia: Funciones de ploteo (plot_graph, plot_scores) no definidas. No se generarán gráficos.")
    # except Exception as e:
    #    print(f"\nError durante el ploteo: {e}")