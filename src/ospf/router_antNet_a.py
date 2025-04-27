import numpy as np
import threading
import logging
from dataclasses import dataclass

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class RouteEntry:
    """Representa una entrada individual en la tabla de enrutamiento."""
    neighbor: int  # ID del nodo vecino (interfaz)
    probability: float  # Probabilidad de selección
    pheromone: float  # Cantidad de feromona
    
    def __str__(self):
        return f"Vecino: {self.neighbor}, Prob: {self.probability:.2f}, Feromona: {self.pheromone}"

class RoutingTable:
    """Clase que representa una tabla de enrutamiento completa para un router."""
    
    def __init__(self):
        """Inicializa una tabla de enrutamiento vacía."""
        self.table = {}  # {destino: [RouteEntry, RouteEntry, ...]}
        self.lock = threading.Lock()
    
    def add_destination(self, destination, entries):
        """Añade un destino con sus entradas de ruta."""
        with self.lock:
            self.table[destination] = entries
    
    def get_entries(self, destination):
        """Obtiene todas las entradas para un destino específico."""
        return self.table.get(destination, [])
    
    def update_entry(self, destination, neighbor, probability=None, pheromone=None):
        """Actualiza una entrada específica para un destino y vecino."""
        with self.lock:
            if destination in self.table:
                for entry in self.table[destination]:
                    if entry.neighbor == neighbor:
                        if probability is not None:
                            entry.probability = probability
                        if pheromone is not None:
                            entry.pheromone = pheromone
                        return True
            return False
    
    def get_all_destinations(self):
        """Retorna todos los destinos en la tabla."""
        return list(self.table.keys())
    
    def __getitem__(self, destination):
        """Permite acceder a las entradas usando la sintaxis: routing_table[destination]"""
        return self.table.get(destination, [])
    
    def __setitem__(self, destination, entries):
        """Permite establecer entradas usando la sintaxis: routing_table[destination] = entries"""
        with self.lock:
            self.table[destination] = entries
    
    def __contains__(self, destination):
        """Permite usar 'destination in routing_table'"""
        return destination in self.table
    
    def items(self):
        """Permite iterar sobre la tabla como un diccionario."""
        return self.table.items()
    
    def __str__(self):
        """Representación en string de la tabla de enrutamiento."""
        result = []
        for dest, entries in self.table.items():
            result.append(f"Destino: {dest}")
            for entry in entries:
                result.append(f"  {entry}")
        return "\n".join(result)
    
    def evap_pheromones(self, rate):
        """Disminuye las feromonas en todas las rutas según la tasa de evaporación."""
        with self.lock:
            for destination, entries in self.table.items():
                for entry in entries:
                    # Aplicar evaporación (reducción de feromonas)
                    entry.pheromone = max(1, entry.pheromone * (1 - rate))
                
                # Recalcular las probabilidades basadas en las nuevas cantidades de feromonas
                sum_pheromones = sum(entry.pheromone for entry in entries)
                for entry in entries:
                    entry.probability = entry.pheromone / sum_pheromones

class RouterAntNet:
    """
    Implementación de un router que utiliza el algoritmo AntNet para el enrutamiento.
    AntNet es un algoritmo bioinspirado basado en colonias de hormigas para el enrutamiento adaptativo.
    """
                
    def __init__(self, node_id, grafo, evaporation_rate=0.05):
        """
        Inicializa un router AntNet.
        
        Args:
            node_id: Identificador único del router
            grafo: Grafo que representa la topología de la red
            evaporation_rate: Tasa de evaporación de feromonas (por defecto 0.05)
        """
        self.node_id = node_id
        self.grafo = grafo
        self.topology_database = {}  # Una copia local de la topología de red
        self.routing_table = RoutingTable()  
        self.lock = threading.Lock()
        self.evaporation_rate = evaporation_rate  # Tasa de evaporación de feromonas
        self._initialize_routing_table()  # Inicializar la tabla de enrutamiento
    
    def _initialize_routing_table(self):
        """
        Inicializa la tabla de enrutamiento del router con sus conexiones adyacentes.
        Para las conexiones no adyacentes se utiliza una distribución de probabilidad
        basada en el peso de las aristas.
        """
        graph = self.grafo
        node = self.node_id
        neighbors = list(graph.neighbors(node))
        
        # Obtener los pesos de las aristas que conectan el nodo con sus vecinos
        weights = [1/graph[node][neighbor]['weight'] for neighbor in neighbors]
        suma_pesos = sum(weights)
        probs = [weight/suma_pesos for weight in weights]
        
        # Inicializar la tabla de enrutamiento con una distribución de probabilidad
        # que da más probabilidad a los nodos con menor peso.
        # Se inicializa el valor de feromona en 100
        routing_sub_table = [RouteEntry(neighbor, prob, 100) for neighbor, prob in zip(neighbors, probs)]

        for nodo in graph.nodes():
            if nodo not in neighbors and nodo != node:
                self.routing_table[nodo] = routing_sub_table
            else:
                self.routing_table[nodo] = [RouteEntry(nodo, 1.0, 100)]  # Probabilidad 1 para el nodo mismo (nodo vecino)  
            
    def update_topology_database(self, grafo):
        """
        Simula la recepción y procesamiento de LSAs (Link State Advertisements).
        """
        self.topology_database = grafo
    
    def calculate_shortest_paths(self, routers, no_ants=20, alpha=0.5, beta=0.5, p_factor=10, no_elite_ants=5):
        """
        Ejecuta el algoritmo de AntColony para hallar las probabilidades de elección de los nodos.
        
        Args:
            routers: Diccionario de routers en la red
            no_ants: Número de hormigas normales
            alpha: Factor que controla la importancia del peso de las aristas
            beta: Factor que controla la importancia de las feromonas
            p_factor: Factor que controla la cantidad de feromona a depositar
            no_elite_ants: Número de hormigas élite
        """
        self.antNet(no_ants, alpha, beta, p_factor, no_elite_ants, routers)

    def get_routing_table(self):
        """
        Devuelve la tabla de enrutamiento del router.
        """
        return self.routing_table

    def get_next_hop(self, destination, visited, probados):
        """
        Obtiene el siguiente salto para alcanzar un destino.
        
        Args:
            destination: Nodo destino
            visited: Conjunto de nodos ya visitados
            probados: Conjunto de nodos ya probados
            
        Returns:
            El ID del siguiente nodo a visitar o None si no hay ruta disponible
        """
        if destination in self.routing_table:
            entries = self.routing_table[destination]
            probs = [x.probability for x in entries if x.neighbor not in visited and x.neighbor not in probados]
            nodes = [x.neighbor for x in entries if x.neighbor not in visited and x.neighbor not in probados]
            if len(nodes) == 0:
                return None
            suma = sum(probs)
            probs = [prob/suma for prob in probs]
            next_node = np.random.choice(nodes, p=probs)
            visited.add(next_node)
            return next_node
        return None
        
    def print_routing_table(self):
        """Muestra la tabla de enrutamiento en un formato legible."""
        print(f"\nTabla de enrutamiento para el router {self.node_id}:")
        print("-" * 60)
        print(f"{'Destino':<10} | {'Siguiente Salto':<15} | {'Probabilidad de elección':<10} | {'Feromonas actuales':<10}")
        print("-" * 60)

        for dest, entries in self.routing_table.items():
            print(f"{dest:<10}")
            for entry in entries:
                print(f"{'':<10} | {entry.neighbor:<15} | {round(entry.probability, 2):<24} | {entry.pheromone:<10}")
            print()
        print("-" * 60)

    def evap_pheromones(self):
        """Disminuye las feromonas en todas las rutas según la tasa de evaporación."""
        self.routing_table.evap_pheromones(self.evaporation_rate)
    
    def _get_router_routing_table(self, router_name, routers):
        """
        Obtiene la tabla de enrutamiento de un router específico.
        """
        if router_name in routers:
            return routers[router_name].get_routing_table()
        else:
            logging.warning(f"Router {router_name} no encontrado.")
            return None
    
    def _choose_next_node(self, current_node, visited, destination, routers):
        """
        Elige el siguiente nodo a visitar en función de las probabilidades de la tabla de enrutamiento.
        
        Args:
            current_node: Nodo actual
            visited: Conjunto de nodos ya visitados
            destination: Nodo destino
            routers: Diccionario de routers en la red
            
        Returns:
            El ID del siguiente nodo a visitar o None si no hay ruta disponible
        """
        routing_table_actual = self._get_router_routing_table(current_node, routers)
        if routing_table_actual is None:
            return None
            
        nodes = [x.neighbor for x in routing_table_actual[destination] if x.neighbor not in visited]
        if len(nodes) == 0:
            return None
            
        probs_originales = [x.probability for x in routing_table_actual[destination] if x.neighbor not in visited]
        suma = sum(probs_originales)
        probs = [prob/suma for prob in probs_originales]
        
        return np.random.choice(nodes, p=probs)
    
    def _update_pheromone(self, path, alpha, beta, pheromone_factor, elite=False, umbral=500):
        """
        Actualiza la cantidad de feromona en las rutas según el camino recorrido.
        
        Args:
            path: Lista de nodos que forman el camino
            alpha: Factor que controla la importancia del peso de las aristas
            beta: Factor que controla la importancia de las feromonas
            pheromone_factor: Factor que controla la cantidad de feromona a depositar
            elite: Indica si la actualización es de una hormiga élite
            umbral: Umbral de feromona para las hormigas élite
        """
        
        routing_table = self.routing_table
        
        with self.lock:
            if len(path) > 2:
                interfaz_camino = path[1]
                for i in range(len(path) - 2):
                    current_node = path[i + 2]
                    for entry in routing_table[current_node]:
                        if entry.neighbor == interfaz_camino:
                            if not elite:
                                entry.pheromone += pheromone_factor
                            else:
                                if entry.pheromone < umbral:
                                    entry.pheromone += pheromone_factor * 10

            self._update_probabilities(path, alpha, beta)
    
    def _update_probabilities(self, path, alpha, beta):
        """
        Actualiza las probabilidades en la tabla de enrutamiento según los pesos y las feromonas.
        
        Args:
            path: Lista de nodos que forman el camino
            alpha: Factor que controla la importancia del peso de las aristas
            beta: Factor que controla la importancia de las feromonas
        """
        routing_table = self.routing_table
        graph = self.grafo
        node = self.node_id
        
        vecinos = list(graph.neighbors(node))
        pesos = [(1/graph[node][neighbor]['weight'])**alpha for neighbor in vecinos]
        
        for i in range(len(path)-2):
            feromonas = [entry.pheromone**beta for entry in routing_table[path[i+2]]]                
            pesos2 = [peso*feromona for peso, feromona in zip(pesos, feromonas)]
            interfaces = [entry.neighbor for entry in routing_table[path[i+2]]]
            
            suma_pesos = sum(pesos2)
            routing_table[path[i+2]] = [
                RouteEntry(
                    interfaces[j], 
                    pesos2[j]/suma_pesos, 
                    routing_table[path[i+2]][j].pheromone
                ) for j in range(len(interfaces))
            ]
    
    def _run_ant(self, is_elite=False, alpha=0.5, beta=0.5, p_factor=10, routers=None):
        """
        Ejecuta una hormiga que busca caminos en la red.
        
        Args:
            is_elite: Indica si es una hormiga élite
            alpha: Factor que controla la importancia del peso de las aristas
            beta: Factor que controla la importancia de las feromonas
            p_factor: Factor que controla la cantidad de feromona a depositar
            routers: Diccionario de routers en la red
            
        Returns:
            El camino encontrado por la hormiga
        """
        graph = self.grafo
        node = self.node_id
        
        # Elegir un destino aleatorio para la hormiga
        if is_elite:
            destination = np.random.choice([n for n in graph.nodes() if n != node])
        else:
            destination = np.random.choice([n for n in graph.nodes() if n != node and n not in graph.neighbors(node)])
        
        # Inicializar la ruta de la hormiga
        path = [node]
        visited = set([node])
        actual_node = node
        
        # Iterar hasta alcanzar el destino o un camino sin salida
        while path[-1] != destination:
            next_node = self._choose_next_node(actual_node, visited, destination, routers)
            
            if next_node is None:
                log_msg = f"No hay camino {'elite ' if is_elite else ''}encontrado de {node} a {destination}"
                logging.info(log_msg)
                logging.debug(f"Nodos visitados: {visited}")
                
                # Añadir feromonas aunque no se haya llegado al destino
                self._update_pheromone(path, alpha, beta, p_factor, is_elite)
                break
                
            actual_node = next_node
            path.append(next_node)
            visited.add(next_node)
        
        # Si se ha llegado al destino, dejar una feromona positiva
        if path[-1] == destination:
            log_msg = f"Camino {'elite ' if is_elite else ''}encontrado de {node} a {destination}"
            logging.info(log_msg)
            logging.debug(f"Nodos visitados: {visited}")
            self._update_pheromone(path, alpha, beta, p_factor, is_elite)
        
        return path

    def antNet(self, no_ants, alpha, beta, p_factor, no_elite_ants, routers):
        """
        Implementa el algoritmo AntNet para actualizar la tabla de enrutamiento.
        
        Args:
            no_ants: Número de hormigas normales
            alpha: Factor que controla la importancia del peso de las aristas
            beta: Factor que controla la importancia de las feromonas
            p_factor: Factor que controla la cantidad de feromona a depositar
            no_elite_ants: Número de hormigas élite
            routers: Diccionario de routers en la red
        """
        # Aplicar evaporación de feromonas antes de enviar nuevas hormigas
        self.evap_pheromones()
        
        # Crear y ejecutar las hormigas normales
        ants = []
        for _ in range(no_ants):
            ant = threading.Thread(
                target=self._run_ant, 
                args=(False, alpha, beta, p_factor, routers)
            )
            ants.append(ant)
            ant.start()
        
        # Crear y ejecutar las hormigas élite
        elite_ants = []
        for _ in range(no_elite_ants):
            elite_ant = threading.Thread(
                target=self._run_ant, 
                args=(True, alpha, beta, p_factor, routers)
            )
            elite_ants.append(elite_ant)
            elite_ant.start()
        
        # Esperar a que terminen todas las hormigas
        for ant in ants:
            ant.join()
        for ant in elite_ants:
            ant.join()







