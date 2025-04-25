from .router_ospf import RouterOSPF

# Clase OSPF para simular el protocolo OSPF en una red
class Ospf:
    def __init__(self, grafo):
        self.grafo = grafo
        self.routers = {}

    # Función para simular OSPF en toda la red
    def simulate_ospf(self):
        """Simula el proceso OSPF en toda la red"""
        routers = {}
        
        # Crear un router para cada nodo
        for node in self.grafo.nodes():
            routers[node] = RouterOSPF(node, self.grafo)
        
        # Cada router actualiza su base de datos topológica y calcula rutas
        for router in routers.values():
            router.update_topology_database(self.grafo)
            router.calculate_shortest_paths()
        
        return routers