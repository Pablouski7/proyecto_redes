from .router_antNet_a import RouterAntNet
import threading
class AntNet:
    def __init__(self, grafo, no_ants=20, alpha=0.5, beta=0.5, p_factor=10, no_elite_ants=5, evaporation_rate=0.1):
        self.grafo = grafo
        self.routers = {}
        self.no_ants = no_ants
        self.alpha = alpha  
        self.beta = beta
        self.p_factor = p_factor
        self.no_elite_ants = no_elite_ants
        self.evaporation_rate = evaporation_rate

    # Función para simular ACO en toda la red
    def simulate_antNet(self):
        def simulate_ant_router(router, self):
            """Simula el proceso ACO en un router específico"""
            # Actualizar la base de datos de topología
            router.update_topology_database(self.grafo)
            
            # Calcular las rutas más cortas
            router.calculate_shortest_paths(self.routers, 
                                            no_ants=self.no_ants, 
                                            alpha=self.alpha, 
                                            beta=self.beta, 
                                            p_factor=self.p_factor, 
                                            no_elite_ants=self.no_elite_ants)
            
            print(f"Router {router.node_id} ha terminado de calcular rutas.")

        routers = {}
        
        # Crear un router para cada nodo
        for node in self.grafo.nodes():
            routers[node] = RouterAntNet(node, self.grafo, self.evaporation_rate)
        self.routers = routers
        # for router in routers.values():
        #     router.print_routing_table()
        #Ejecutar cada router de manera linear
        
        # for router in routers.values():
        #     simulate_ant_router(router, self)
        #     break
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
