from .router_antNet_a import RouterAntNet
import threading
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
            router.calculate_shortest_paths(self.routers)
            
            print(f"Router {router.node_id} ha terminado de calcular rutas.")

        routers = {}
        
        # Crear un router para cada nodo
        for node in self.grafo.nodes():
            routers[node] = RouterAntNet(node, self.grafo)
        self.routers = routers
        #Crear un hilo para cada router
        threads = []
        for router in routers.values():
            thread = threading.Thread(target=simulate_ant_router, args=(router,self))
            threads.append(thread)
            thread.start()
            # break
        # Esperar a que todos los hilos terminen
        for thread in threads:
            thread.join()
        
        return routers
