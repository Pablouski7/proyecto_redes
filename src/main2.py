import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import random
from ospf.antNet import AntNet
from topologia import Topologia

# Función principal para ejecutar todo
def main():
    topologia1 = Topologia()
    # Crear grafo de prueba
    grafo, aristas_eliminadas = topologia1.crear_red(7, 7, 7)
    
    # Imprimir información del grafo generado
    print("Grafo generado:")
    print(grafo.nodes())
    print("Aristas eliminadas:")
    print(aristas_eliminadas)
    
    # Imprimir las latitudes y longitudes de los nodos
    for nodo in grafo.nodes(data=True):
        print(f"Nodo: {nodo[0]}, Latitud: {nodo[1]['lat']}, Longitud: {nodo[1]['lon']}")
    
    # Imprimir los pesos de las aristas
    print("Pesos de las aristas (tiempo de demora en ms):")
    for u, v, data in grafo.edges(data=True):
        print(f"Arista: {u} - {v}, Peso: {data['weight']} ms")
    
    # Visualizar el grafo original (como en el script original)
    plt.figure(figsize=(10, 6))
    nx.draw(grafo, with_labels=True, node_size=700, node_color='lightblue', 
            font_size=10, font_weight='bold')
    plt.title("Grafo generado")
    plt.show()
    
    antNet_run = AntNet(grafo)
    # Simular OSPF
    
    routers = antNet_run.simulate_antNet()
    # for router in routers.values():
    #     #imprimir las tablas de los router que tienen valores mayores a 100
    #     #en la feromona
    #     tabla=router.routing_table
    #     for dest, states in tabla.items():
    #         for state in states:
    #             if state[2] > 100:
    #                 router.print_routing_table()
    #                 break
    
    # Seleccionar origen y destino para trazar una ruta
    all_nodes = list(grafo.nodes())
    source = None
    destination = None
    
    # Intentar seleccionar nodos primarios y secundarios de diferentes segmentos
    for node in all_nodes:
        if node.startswith("SS0_") or node.startswith("P0"):
            source = node
        elif node.startswith("SS4_") or node.startswith("P4"):
            destination = node
    
    # Si no se encontraron los nodos específicos, seleccionar los primeros disponibles
    if source is None and len(all_nodes) > 0:
        source = all_nodes[0]
    if destination is None and len(all_nodes) > 1:
        destination = all_nodes[-1]
        
    # Asegurarse de que origen y destino sean diferentes
    if source == destination and len(all_nodes) > 1:
        destination = all_nodes[1]
    
    if source is not None and destination is not None:
        print(f"\nRuta de {source} a {destination}")
        path = topologia1.trace_route(source, destination, routers)
        print(f"Ruta calculada: {path}")

        # Comparar con la implementación de NetworkX para verificar
        try:
            nx_path = nx.shortest_path(grafo, source=source, target=destination, weight='weight')
            print(f"Ruta calculada por NetworkX: {nx_path}")
            if path == nx_path:
                print("¡Las rutas coinciden! La implementación es correcta.")
            else:
                print("Las rutas no coinciden. Puede haber un problema en la implementación.")
        except nx.NetworkXNoPath:
            print("NetworkX no encontró una ruta.")
        topologia1.plot(path)

if __name__ == "__main__":
    main()