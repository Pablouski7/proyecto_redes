from .router_antNet import RouterAntNet
import threading

# Clase OSPF para simular el protocolo OSPF en una red
class AntNet:

    def __init__(self, grafo):
        self.grafo = grafo
        self.routers = {}

    # Función para simular ACO en toda la red
    def simulate_antNet(self):
        def simulate_ant_router(router, self):
            """Simula el proceso ACO en un router específico"""
            # Actualizar la base de datos de topología
            router.update_topology_database(self.grafo)
            
            # Calcular las rutas más cortas
            router.calculate_shortest_paths()
            print(f"Router {router.node_id} ha terminado de calcular rutas.")

            
        """Simula el proceso OSPF en toda la red"""
        routers = {}
        
        # Crear un router para cada nodo
        for node in self.grafo.nodes():
            routers[node] = RouterAntNet(node, self.grafo)
        #Crear un hilo para cada router
        threads = []
        for router in routers.values():
            thread = threading.Thread(target=simulate_ant_router, args=(router,self))
            threads.append(thread)
            thread.start()
        # Esperar a que todos los hilos terminen
        for thread in threads:
            thread.join()
        
        return routers
