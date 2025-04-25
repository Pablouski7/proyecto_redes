import heapq

# Clase Router para simular el comportamiento de OSPF
class RouterOSPF:
    def __init__(self, node_id, grafo):
        self.node_id = node_id
        self.grafo = grafo
        self.routing_table = {}  # {destino: (next_hop, costo)}
        self.topology_database = {}  # Una copia local de la topología de red
        
    def update_topology_database(self, grafo):
        """Simula la recepción y procesamiento de LSAs (Link State Advertisements)"""
        self.topology_database = grafo
        
    def calculate_shortest_paths(self):
        """Implementación de Dijkstra para calcular las rutas más cortas a todos los destinos"""
        distances = {node: float('infinity') for node in self.grafo.nodes()}
        predecessors = {node: None for node in self.grafo.nodes()}
        distances[self.node_id] = 0
        pq = [(0, self.node_id)]
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # Si ya encontramos un camino más corto, ignoramos
            if current_distance > distances[current_node]:
                continue
                
            # Explorar vecinos
            for neighbor in self.grafo.neighbors(current_node):
                weight = self.grafo[current_node][neighbor]['weight']
                distance = current_distance + weight
                
                # Si encontramos un camino más corto a 'neighbor'
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        # Construir tabla de enrutamiento
        for dest_node in self.grafo.nodes():
            if dest_node == self.node_id:
                continue
                
            # Reconstruir el camino hacia el destino
            path = []
            current = dest_node
            while current != self.node_id:
                path.append(current)
                current = predecessors[current]
                if current is None:  # No hay camino
                    break

            if path:
                path.reverse()
                next_hop = path[0]
                self.routing_table[dest_node] = (next_hop, distances[dest_node])
            else:
                self.routing_table[dest_node] = (None, float('infinity'))
                
    def get_routing_table(self):
        """Devuelve la tabla de enrutamiento"""
        return self.routing_table
    
    def get_next_hop(self, destination):
        """Obtiene el siguiente salto para alcanzar un destino"""
        if destination in self.routing_table:
            return self.routing_table[destination][0]
        return None

    def print_routing_table(self):
        """Muestra la tabla de enrutamiento en un formato legible"""
        print(f"\nTabla de enrutamiento para el router {self.node_id}:")
        print("-" * 60)
        print(f"{'Destino':<10} | {'Siguiente Salto':<15} | {'Costo (ms)':<10}")
        print("-" * 60)

        for dest, (next_hop, cost) in sorted(self.get_routing_table().items()):
            print(f"{dest:<10} | {str(next_hop):<15} | {cost:<10.4f}")
