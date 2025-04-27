import numpy as np
import threading
       
class RouterAntNet:
                
    def __init__(self, node_id, grafo):
        def iniciar_routing_table(self):
            # Inicizaliar la routing tables del router actual
            # Con sus conexiones adyacentes
            # A las conexiones no adyacentes se las maneja por medio de una 
            # distribución de probabilidad considerando el peso de la arista
            graph = self.grafo
            node = self.node_id
            neighbors = graph.neighbors(node)
            neighbors = list(neighbors)
            # print(f"Vecinos de {node}: {neighbors}")
            #Obtener los pesos de las aristas que conectan el nodo con sus vecinos
            weights = [1/graph[node][neighbor]['weight'] for neighbor in neighbors]
            suma_pesos = sum(weights)
            probs = [weight/suma_pesos for weight in weights]
            #Inicializar la tabla de enrutamiento con una distribución de probabilidad que de 
            #más probabilidad a los nodos con menor peso. Los pesos que ingresan 
            #a la distribucion de probabilidad son los inversos de los pesos de las aristas
            #que conectan el nodo con sus vecinos (a menor peso, mayor probabilidad)
            #Se inicializa el valor de feromona en 100
            
            routing_sub_table = [[neighbor, prob,100] for neighbor, prob in zip(neighbors, probs)]

            for nodo in graph.nodes():
                if nodo not in neighbors and nodo!=node:
                    self.routing_table[nodo] = routing_sub_table
                else:
                    self.routing_table[nodo] = [[nodo, 1.0,100]] # Probabilidad 1 para el nodo mismo (nodo vecino)   
            # print("Routing table inicializada")
            # print(self.routing_table)
        #Funcion anidada para inicializar la tabla de enrutamiento
        self.node_id = node_id
        self.grafo = grafo
        self.topology_database = {}  # Una copia local de la topología de red
        self.routing_table = {}
        self.lock = threading.Lock()
        #Locks a nivel de clase ya que cada router unicamente modifica su propia tabla de enrutamiento
        iniciar_routing_table(self)  # {destino: (next_hop, costo)}
    


    def update_topology_database(self, grafo):
        """Simula la recepción y procesamiento de LSAs (Link State Advertisements)"""
        self.topology_database = grafo
    
    def calculate_shortest_paths(self,routers,no_ants=20,alpha=0.5,beta=0.5, PFactor=10,no_elite_ants=5):
        #Como se ejecutan hilos es importante manejar correctamente el procceso de concurrencia
        #y sincronización de los hilos al actualizar la tabla de enrutamiento
        """Ejecuta el algoritmo de AntColony para hallar las probabilidades de elección de los nodos"""
        #Ejecutar antNet
        self.antNet(no_ants,alpha,beta,PFactor,no_elite_ants,routers)

    def get_routing_table(self):
        """Devuelve la tabla de enrutamiento"""
        return self.routing_table

    def get_next_hop(self, destination, visited, probados):
        """Obtiene el siguiente salto para alcanzar un destino"""
        if destination in self.routing_table:
            
            
            probs=[x[1] for x in self.routing_table[destination] if x[0] not in visited and x[0] not in probados]
            nodes=[x[0] for x in self.routing_table[destination] if x[0] not in visited and x[0] not in probados]
            if(len(nodes)==0):
                return None
            suma=sum(probs)
            probs=[prob/suma for prob in probs]
            next_node=np.random.choice(nodes, p=probs)
            visited.add(next_node)
            return next_node
        return None
           

    def print_routing_table(self):
        """Muestra la tabla de enrutamiento en un formato legible"""
        print(f"\nTabla de enrutamiento para el router {self.node_id}:")
        print("-" * 60)
        print(f"{'Destino':<10} | {'Siguiente Salto':<15} | {'Probabilidad de elección':<10} | {'Feromonas actuales':<10}")
        print("-" * 60)

        for dest, states in self.routing_table.items():
            print(f"{dest:<10}")
            for state in states:
                print(f"{'':<10} | {state[0]:<15} | {round(state[1], 2):<24} | {state[2]:<10}")
            print()
        print("-" * 60)


    def antNet (self,no_ants,alpha,beta, PFactor,no_elite_ants,routers):
        graph=self.grafo
        routing_table=self.routing_table
        node=self.node_id
        #Definicion de funciones anidadas
        def updatePheromone(routing_table, node, alpha, beta, path,graph, pheromone_factor, elite=False, umbral=500):
            #Actualizar la tabla de enrutamiento del nodo actual
            #Aumentar la probabilidad de elección de los nodos vecinos
            #De acuerdo a la cantidad de feromona que tienen
            #Obtener la interfaz por la que se accede a ese destino
            #El segundo nodo de la ruta es la interfaz (conexion directa del nodo actual)
            #Por la que se accede a ese destino
            print("Camino a actualizar: ", path)
            print("Añadiendo feromonas")
            
            #Actualizar la probabilidad de elección de los nodos vecinos
            #De acuerdo a la cantidad de feromona que tienen y al peso de la arista
            #Se suma por defecto el pheromone_factor a la probabilidad de elección de los nodos vecinos
            #los nodos del path ya que aunque no se haya llegado al destino, se ha recorrido un camino
            #y se encontraron indirectamente rutas hacia otras partes de la red
            #Si la hormiga es élite realiza un aumento fuerte para los nodos alcanzados
            # (factor de 10 vecces mayor que una hormiga normal)
            #Pero únicamente realiza este aumento si hay poca feromona en el nodo destino de la tabla de enrutamiento
            with self.lock:
                if(len(path)>2):
                    interfaz_camino=path[1]
                    for i in range(len(path)-2):
                        for state in routing_table[path[i+2]]:
                            if state[1]!=1 and state[0]== interfaz_camino:
                                if not elite:
                                    state[2] += pheromone_factor
                                else:
                                    if state[2] < umbral:
                                        state[2] += pheromone_factor*10

                updateProbs(routing_table, node, alpha, beta, path, graph)

            # Clase Router para simular el comportamiento de OSPF
        def get_routing_table(router_name, routers):
            """Devuelve la tabla de enrutamiento del router"""
            if router_name in routers:
                return routers[router_name].get_routing_table()
            else:
                print(f"Router {router_name} no encontrado.")
                return None   
                    
                    
        def updateProbs(routing_table, node, alpha, beta, path, graph):
             #Actualizar la probabilidad obteniendo los pesos de los nodos vecinos
            #Y sumar la cantidad de feromona que tiene cada uno de ellos
            vecinos = graph.neighbors(node)
            vecinos = list(vecinos)
            
            pesos = [(1/graph[node][neighbor]['weight'])**alpha for neighbor in vecinos]
            
            for i in range(len(path)-2):
                # print("Nodo actual: ", path[i+2])
                # print("Vecinos: ", vecinos)
                feromonas= [state[2]**beta for state in routing_table[path[i+2]]]                
                
                pesos2= [peso*feromona for peso, feromona in zip(pesos, feromonas)]
                
                interfaces= [state[0] for state in routing_table[path[i+2]]]
                
                suma_pesos = sum(pesos2)
                # print("Suma de pesos: ", suma_pesos)
                # print("Pesos: ", pesos)
                # print("Pesos2: ", pesos2)
                # print("Feromonas: ", feromonas)
                #Actualizar la probabilidad de elección de los nodos vecinos para este nodo destino
                routing_table[path[i+2]] = [[interfaces[j], pesos[j]/suma_pesos, routing_table[path[i+2]][j][2]] for j in range(len(interfaces))]
                
                    
            
        
        
        def choose_next_node(node, routing_table,visited,destination,routers):
            #Obtener la routing table del nodo actual, para el destino seleccionado
            routing_table_actual= get_routing_table(node, routers)
            if routing_table_actual is None:
                return None
            nodes=[x[0] for x in routing_table_actual[destination] if x[0] not in visited]
            # print("Nodos disponibles: ", nodes)
            # print("Nodo actual: ", node)    
            #Calcular las nuevas probabilidades considerando unicamente los nodos no visitados
            if len(nodes)==0:
                return None
            probs_originales=[x[1] for x in routing_table_actual[destination] if x[0] not in visited]
            # print("Probabilidades originales: ", probs_originales)
            probs=[prob/sum(probs_originales) for prob in probs_originales]
            # print("Probabilidades normalizadas: ", probs)
            #Utilizar la función random.choice para elegir el siguiente nodo a visitar
            next_node=np.random.choice(nodes, p=probs)
            return next_node
        def Ant(node, graph, alpha, beta, routing_table, PFactor,routers):
            #Elegir un destino aleatorio para la hormiga, el destino no puede ser el nodo actual
            #Ni tampoco los vecinos del nodo actual ya que de ellos se conoce la ruta
            destination = np.random.choice([n for n in graph.nodes() if n != node and n not in graph.neighbors(node)])
            #Inicializar la ruta de la hormiga, crear un set para los nodos visitados
            #Y así evitar ciclos
            path = [node]
            visited=set()
            visited.add(node)
            actual_node=node
            #Iterar la hormiga hasta alcanzar el destino o hasta que se llegue a un camino sin salida
            while path[-1] != destination:
                #Elegir el siguiente nodo a visitar
                # print("Nodo actual: ", actual_node)
                # print("Destino: ", destination)
                # print("Nodos visitados: ", visited)
                # print("Ruta: ", path)
                next_node = choose_next_node(actual_node, routing_table, visited, destination, routers)
                #Si no hay nodos disponibles, salir del bucle
                if next_node is None:
                    print("No hay camino encontrado de " + str(node) + " a " + str(destination))
                    print("Nodos visitados: ", visited)
                    #Añadir feromonas en la tabla de enrutamiento del nodo actual
                    #a los nodos visitados aunque no se haya llegado al destino
                    updatePheromone(routing_table, node, alpha, beta, path, graph, PFactor)
                    break
                #Agregar el nodo a la ruta y marcarlo como visitado
                actual_node=next_node
                path.append(next_node)
                visited.add(next_node)
                
            if path[-1] == destination:
                #Si se ha llegado al destino, dejar una feromona positiva en la ruta
                #Para cada nodo en la ruta aumentar segun la feromona su probabilidad de elección
                #La actualización solo se realizaría en la tabla de enrutamiento del nodo actual
                #Para así evitar problemas de concurrencia
                print("Camino encontrado de " + str(node) + " a " + str(destination))
                print("Nodos visitados: ", visited)
                updatePheromone(routing_table, node, alpha, beta, path, graph, PFactor)
        def eliteAnt(node, graph, alpha, beta, routing_table, PFactor,routers):
            #Elegir un destino aleatorio para la hormiga, el destino no puede ser el nodo actual
            destination = np.random.choice([n for n in graph.nodes() if n != node])
            #Inicializar la ruta de la hormiga, crear un set para los nodos visitados
            #Y así evitar ciclos
            path = [node]
            visited=set()
            visited.add(node)
            actual_node=node
            #Iterar la hormiga hasta alcanzar el destino o hasta que se llegue a un camino sin salida
            while path[-1] != destination:
                #Elegir el siguiente nodo a visitar
                next_node = choose_next_node(actual_node, routing_table, visited, destination, routers)
                #Si no hay nodos disponibles, salir del bucle
                if next_node is None:
                    print("No hay camino elite encontrado de " + str(node) + " a " + str(destination))
                    print("Nodos visitados: ", visited)
                    #Añadir feromonas en la tabla de enrutamiento del nodo actual
                    #a los nodos visitados aunque no se haya llegado al destino
                    updatePheromone(routing_table, node, alpha, beta, path, graph, PFactor, True)
                    break
                #Agregar el nodo a la ruta y marcarlo como visitado
                actual_node=next_node
                path.append(next_node)
                visited.add(next_node)
                
            if path[-1] == destination:
                print("Camino elite encontrado de " + str(node) + " a " + str(destination))
                print("Nodos visitados: ", visited)
                #Si se ha llegado al destino, dejar una feromona positiva en la ruta
                #Para cada nodo en la ruta aumentar segun la feromona su probabilidad de elección
                #La actualización solo se realizaría en la tabla de enrutamiento del nodo actual
                #Para así evitar problemas de concurrencia
                updatePheromone(routing_table, node, alpha, beta, path, graph, PFactor)
                

        #Crear un hilo para cada hormiga
        ants = []
        for i in range(no_ants):
            ant = threading.Thread(target=Ant, args=(node, graph, alpha, beta, routing_table, PFactor,routers))
            ants.append(ant)
            ant.start()
        #Crear un hilo para cada hormiga élite
        elite_ants = []
        for i in range(no_elite_ants):
            elite_ant = threading.Thread(target=eliteAnt, args=(node, graph, alpha, beta, routing_table, PFactor,routers))
            elite_ants.append(elite_ant)
            elite_ant.start()
        # Iniciar todos los hilos y esperar a que terminen
        for ant in ants:
            ant.join()
        for ant in elite_ants:
            ant.join()




            
            

    