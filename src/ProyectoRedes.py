#Importar libreria para uso de grafos
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import random
#Función para obtener la distancia esférica a partir de 
# la latitud y longitud de dos puntos
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en km

    # Convertir grados a radianes
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    d = R * c  # distancia en km
    return d
def crearGrafoPrueba(no_seg_sec, maximo_nodos_sec, ratio_desviacion=0.05):
    #Función que crea un grafo simple de n segmentos secundarios y n segmentos primarios
    #Los segmentos secundarios únicamente se conectan a los segmentos primarios y los segmentos primarios se conectan entre sí
    #Cada segmento secundario contiene un número aleatorio de nodos entre 1 y maximo_nodos_sec
    #Los semgentos primarios son cada uno un nodo y se conectan entre sí
    #Las conexiones de los segmentos primarios se realiza primero una conexión fully connected entre ellos
    #Y después se eliminan aleatoriamente ciertas conexiones
    #Guardando el estado de que arista (nodo1, nodo2) se eliminó para evitar generar un grafo no conexo
    #Ratio_desviacion es el porcentaje de desviación de latitud y longitud maximo de los segmentos secundarios respecto a su
    #segmento primario con el que se conecta
    
    # Crear un grafo vacío
    G = nx.Graph()
    # Crear segmentos secundarios y primarios
    #Crear segmentos primarios y sus valores aleatorios de latitud y longitud
    for i in range(no_seg_sec):
        # Crear un nodo para el segmento primario
        G.add_node(f"P{i}", lat=random.uniform(-90, 90), lon=random.uniform(-180, 180))
    for i in range(no_seg_sec):
        # Crear un segmento secundario con un número aleatorio de nodos entre 1 y maximo_nodos_sec
        num_nodos_sec = random.randint(1, maximo_nodos_sec)
        nodos_sec = [f"SS{i}_{j}" for j in range(num_nodos_sec)]
        #Latitud y longitud de cada nodo secundario
        nodos_sec_lat_lon = []
        max_lat = G.nodes[f"P{i}"]['lat'] + ratio_desviacion * random.uniform(-1, 1)
        min_lat = G.nodes[f"P{i}"]['lat'] - ratio_desviacion * random.uniform(-1, 1)
        max_lon = G.nodes[f"P{i}"]['lon'] + ratio_desviacion * random.uniform(-1, 1)
        min_lon = G.nodes[f"P{i}"]['lon'] - ratio_desviacion * random.uniform(-1, 1)
        for j in range(num_nodos_sec):
            lat = random.uniform(min_lat, max_lat)
            lon = random.uniform(min_lon, max_lon)
            nodos_sec_lat_lon.append((lat, lon))
            # Añadir el nodo secundario al grafo con su latitud y longitud
            G.add_node(f"SS{i}_{j}", lat=lat, lon=lon)
        G.add_nodes_from(nodos_sec)

        # Conectar los nodos del segmento secundario al nodo primario correspondiente
        for nodo in nodos_sec:
            G.add_edge(nodo, f"P{i}")
    
    # Conectar los segmentos primarios entre sí (fully connected)
    for i in range(no_seg_sec):
        for j in range(i + 1, no_seg_sec):
            G.add_edge(f"P{i}", f"P{j}")
    
    # Eliminar aleatoriamente algunas conexiones entre segmentos primarios
    # Guardar el estado de las aristas eliminadas para evitar generar un grafo no conexo
    aristas_eliminadas = []
    aristas = list(G.edges())
    num_aristas_a_eliminar = random.randint(1, len(aristas) // 2)  # Eliminar hasta la mitad de las aristas
    for _ in range(num_aristas_a_eliminar):
        arista= random.choice(aristas)
        #Si el nodo inicial o el nodo final no son segmentos primarios, no se elimina nada
        if arista[0][0] != "P" or arista[1][0] != "P":
            continue
        #Si el nodo inicial o el nodo final de la arista ya existe en la lista de aristas eliminadas, no se elimina nada
        if (arista[0], arista[1]) not in aristas_eliminadas and (arista[1], arista[0]) not in aristas_eliminadas:
            G.remove_edge(arista[0], arista[1])
            aristas_eliminadas.append((arista[0], arista[1]))
            
    #Calcular la distancia haversine en las aristas del grafo y guardar en el la clave 'distance'
    for u, v in G.edges():
        lat1 = G.nodes[u]['lat']
        lon1 = G.nodes[u]['lon']
        lat2 = G.nodes[v]['lat']
        lon2 = G.nodes[v]['lon']
        distancia = haversine(lat1, lon1, lat2, lon2)
        G[u][v]['distance'] = distancia*1000  # Multiplicar por 1000 para convertir a metros

    # Devolver el grafo creado y las aristas eliminadas
    return G, aristas_eliminadas

def actualizarPesos(grafo):
    #Función que actualiza los pesos de las aristas del grafo
    #Calcula la métrica para hallar la distancia más corta
    #La metrica es de tiempo se resume en: distancia_haversine/velocidad_luz + latencia_actual_conexión . La métrica se usará en milisegundos
    
    #La latencia actual de la conexión es un valor aleatorio entre 10 y 1000 ms
    
    velocidad_luz = 299792458  # Velocidad de la luz en m/s
    for u, v, data in grafo.edges(data=True):
        distancia = data['distance']  # Distancia en metros
        tiempo1 = distancia / velocidad_luz * 1000  # Convertir a milisegundos
        latencia = random.uniform(10, 1000)
        # Actualizar el peso de la arista en el grafo
        grafo[u][v]['weight'] = tiempo1 + latencia
        
    

#Generar un grafo de prueba
grafo, aristas_eliminadas = crearGrafoPrueba(5, 10)
#Imprimir el grafo generado
print("Grafo generado:")
print(grafo.nodes())
print("Aristas eliminadas:")
print(aristas_eliminadas)
#Imprimir las latitudes y longitudes de los nodos
for nodo in grafo.nodes(data=True):
    print(f"Nodo: {nodo[0]}, Latitud: {nodo[1]['lat']}, Longitud: {nodo[1]['lon']}")
#Actualizar los pesos de las aristas
actualizarPesos(grafo)
#Imprimir los pesos de las aristas
print("Pesos de las aristas (tiempo de demora en ms):")
for u, v, data in grafo.edges(data=True):
    print(f"Arista: {u} - {v}, Peso: {data['weight']} ms")
    
#Mostrar el grafo generado
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
nx.draw(grafo, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
plt.title("Grafo generado")
plt.show()
