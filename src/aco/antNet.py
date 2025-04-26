import random
import threading

def antNet (graph,routing_table,node):


    no_ants=50
    alpha=0.5
    beta=0.5

    #Crear un hilo para cada hormiga
    ants = []
    for i in range(no_ants):
        ant = Ant(node, graph, alpha, beta, routing_table)
        ants.append(ant)
        ant.start()
        
    #Iniciar todos los hilos y esperar a que terminen
    for ant in ants:
        ant.join()


    
def Ant(node, graph, alpha, beta, routing_table):
    #Elegir un destino aleatorio para la hormiga, el destino no puede ser el nodo actual
    destination = random.choice([n for n in graph.nodes() if n != node])
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
        next_node = choose_next_node(path[-1], graph, alpha, beta, routing_table)
        #Si no hay nodos disponibles, salir del bucle
        if next_node is None:
            #Dar una ponderación negativa al ultimo nodo visitado,
            #Indicando que ese camino no es bueno para llegar al destino
            #Dejar una especie de feromona negra
            for i in range(len(path)-1):
                for state in routing_table[path[i]]:

                
            
            break
        #Agregar el nodo a la ruta y marcarlo como visitado
        path.append(next_node)
        visited.add(next_node)
        
    if path[-1] == destination:
        #Si se ha llegado al destino, dejar una feromona positiva en la ruta, en la tabla de routing de
        #cada nodo
        
def choose_next_node(node, routing_table,visited):

    nodes=[x[0] for x in routing_table[node] if x[0] not in visited]
    #Calcular las nuevas probabilidades considerando unicamente los nodos no visitados
    if len(nodes)==0:
        return None
    probs_originales=[x[1] for x in routing_table[node] if x[0] not in visited]
    probs=[prob/sum(probs_originales) for prob in probs_originales]
    #Utilizar la función random.choice para elegir el siguiente nodo a visitar
    next_node=random.choice(nodes, p=probs)
    return next_node
    
def update_routing_table(    
    
    
    
    