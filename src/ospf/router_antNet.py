import numpy as np
import threading

# Clase Router para simular el comportamiento de OSPF
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
            weights = [graph[node][neighbor]['weight'] for neighbor in neighbors]
            suma_pesos = sum(weights)
            probs = [weight/suma_pesos for weight in weights]
            #Inicializar la tabla de enrutamiento con una distribución de probabilidad que de 
            #más probabilidad a los nodos con menor peso
            routing_sub_table = [(neighbor, prob,0) for neighbor, prob in zip(neighbors, probs)]

            for nodo in graph.nodes():
                if nodo not in neighbors:
                    self.routing_table[nodo] = routing_sub_table
                elif nodo!= node:
                    self.routing_table[nodo] = [(nodo, 1.0,0)] # Probabilidad 1 para el nodo mismo (nodo vecino)   
            # print("Routing table inicializada")
            # print(self.routing_table)
        #Funcion anidada para inicializar la tabla de enrutamiento
        self.node_id = node_id
        self.grafo = grafo
        self.topology_database = {}  # Una copia local de la topología de red
        self.routing_table = {}
        iniciar_routing_table(self)  # {destino: (next_hop, costo)}
    
 
    def update_topology_database(self, grafo):
        """Simula la recepción y procesamiento de LSAs (Link State Advertisements)"""
        self.topology_database = grafo
    
    def calculate_shortest_paths(self,no_ants=50,alpha=0.5,beta=0.5, PFactor=20,no_elite_ants=5):
        """Ejecuta el algoritmo de AntColony para hallar las probabilidades de elección de los nodos"""
        #Ejecutar antNet
        
        self.antNet(no_ants,alpha,beta,PFactor,no_elite_ants)

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


    def antNet (self,no_ants,alpha,beta, PFactor,no_elite_ants):
        graph=self.grafo
        routing_table=self.routing_table
        node=self.node_id
        #Definicion de funciones anidadas
        def updatePheromone(routing_table, node, alpha, beta, path,graph, pheromone_factor):
            #Actualizar la tabla de enrutamiento del nodo actual
            #Aumentar la probabilidad de elección de los nodos vecinos
            #De acuerdo a la cantidad de feromona que tienen
            #Obtener la interfaz por la que se accede a ese destino
            #El segundo nodo de la ruta es la interfaz (conexion directa del nodo actual)
            #Por la que se accede a ese destino
            interfaz_camino=path[1]
            #Actualizar la probabilidad de elección de los nodos vecinos
            #De acuerdo a la cantidad de feromona que tienen y al peso de la arista
            #Se resta por defecto el pheromone_factor a la probabilidad de elección de los nodos vecinos
            #Si es un pheromone factor positivo, el costo total en la tabla de enrutamiento se reduce
            #Caso contrario, se aumenta (cuando no se llega al destino)
            if(len(path)>2):
                for i in range(len(path)-2):
                    for state in routing_table[path[i+2]]:
                        if state[1]!=1 and state[0]== interfaz_camino:
                            state[2] -= pheromone_factor
            updateProbs(routing_table, node, alpha, beta, path, graph)
           
                    
                    
        def updateProbs(routing_table, node, alpha, beta, path, graph):
             #Actualizar la probabilidad obteniendo los pesos de los nodos vecinos
            #Y sumar la cantidad de feromona que tiene cada uno de ellos
            vecinos = graph.neighbors(node)
            pesos = [graph[node][neighbor]['weight'] for neighbor in vecinos]
            feromonas = [routing_table[neighbor][2] for neighbor in vecinos]
            suma=sum(pesos)+sum(feromonas)
            #A cada nodo del path encontrado se recalcula su estado de probabilidad
            #Considerando la cantidad de feromona que tiene y el peso de la arista
            #de la interfaz en cuestión
            if(len(path)>2):
                for i in range(len(path)-2):
                    for state in routing_table[path[i+2]]:
                        routing_table[path[i+2]][1] = (state[1]+state[2])/suma         
            
        
        
        def choose_next_node(node, routing_table,visited):
            nodes=[x[0] for x in routing_table[node] if x[0] not in visited]
            #Calcular las nuevas probabilidades considerando unicamente los nodos no visitados
            if len(nodes)==0:
                return None
            probs_originales=[x[1] for x in routing_table[node] if x[0] not in visited]
            probs=[prob/sum(probs_originales) for prob in probs_originales]
            #Utilizar la función random.choice para elegir el siguiente nodo a visitar
            next_node=np.random.choice(nodes, p=probs)
            return next_node
        def Ant(node, graph, alpha, beta, routing_table, PFactor):
            #Elegir un destino aleatorio para la hormiga, el destino no puede ser el nodo actual
            destination = np.random.choice([n for n in graph.nodes() if n != node])
            #Inicializar la ruta de la hormiga, crear un set para los nodos visitados
            #Y así evitar ciclos
            path = [node]
            visited=set()
            visited.add(node)
            camino_sin_salida=False
            actual_node=node
            #Iterar la hormiga hasta alcanzar el destino o hasta que se llegue a un camino sin salida
            while path[-1] != destination:
                #Elegir el siguiente nodo a visitar
                next_node = choose_next_node(actual_node, routing_table, visited)
                #Si no hay nodos disponibles, salir del bucle
                if next_node is None:
                    #Dar una ponderación negativa al ultimo nodo visitado,
                    #Indicando que ese camino no es bueno para llegar al destino
                    #Dejar una especie de feromona negra
                    # for i in range(len(path)-1):
                    #     for state in routing_table[path[i]]:
                    updatePheromone(routing_table, node, alpha, beta, path, graph, pheromone_factor=-PFactor)
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
                updatePheromone(routing_table, node, alpha, beta, path, graph, PFactor)
                

        #Crear un hilo para cada hormiga
        ants = []
        for i in range(no_ants):
            ant = threading.Thread(target=Ant, args=(node, graph, alpha, beta, routing_table, PFactor))
            ants.append(ant)
            ant.start()
        #Crear un hilo para cada hormiga élite
        elite_ants = []
        for i in range(no_elite_ants):
            ant = threading.Thread(target=Ant, args=(node, graph, alpha, beta, routing_table, PFactor*10))
            elite_ants.append(ant)
            ant.start()    
        #Iniciar todos los hilos y esperar a que terminen
        for ant in ants:
            ant.join()
        for ant in elite_ants:
            ant.join()


        

            
            

    