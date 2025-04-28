import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import random
from algoritmos.aco_net import AcoNet
from algoritmos.ospf import Ospf
from topologia import Topologia

def create_network(n_segments=7, max_nodes=5, ratio_deviation=30, seed=433):
    """Crear y configurar la topología de red"""
    topologia = Topologia(seed=seed)
    grafo, aristas_eliminadas = topologia.crear_red(n_segments, max_nodes, ratio_deviation)
    return topologia, grafo

def select_test_nodes(grafo, seed=433):
    """Seleccionar nodos origen y destino para pruebas"""
    all_nodes = list(grafo.nodes())
    random.seed(seed)  # Para reproducibilidad
    source, destination = random.sample([node for node in all_nodes if 'SS' in node], 2)
    return source, destination

def trace_and_compare(topologia, grafo, source, destination, router_tables, algorithm_name):
    """Trazar una ruta y compararla con NetworkX"""
    print(f"\n--- Ruta con {algorithm_name} ---")
    print(f"De {source} a {destination}")
    
    # Obtener la ruta calculada por el algoritmo
    path = topologia.trace_route(source, destination, router_tables)
    print(f"Ruta calculada por {algorithm_name}: {path}")
    
    # Comparar con NetworkX
    try:
        nx_path = nx.shortest_path(grafo, source=source, target=destination, weight='weight')
        print(f"Ruta calculada por NetworkX: {nx_path}")
        
        # Comparar longitudes de ruta para evaluar eficiencia
        nx_length = sum(grafo[nx_path[i]][nx_path[i+1]]['weight'] for i in range(len(nx_path)-1))
        if path:
            path_length = sum(grafo[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            print(f"Longitud de ruta {algorithm_name}: {path_length:.2f}")
            print(f"Longitud de ruta NetworkX: {nx_length:.2f}")
            print(f"Diferencia: {path_length - nx_length:.2f}")
        
        if path == nx_path:
            print("✓ Las rutas coinciden exactamente.")
        else:
            print("⚠ Las rutas son diferentes.")
    except nx.NetworkXNoPath:
        print("✗ NetworkX no encontró una ruta.")
    
    return path

def main():
    # Crear y mostrar la topología de red
    topologia, grafo = create_network(n_segments=5, max_nodes=7, ratio_deviation=30, seed=103)
    topologia.plot()
    
    # Ejecutar simulaciones de los algoritmos
    print("\n==== Ejecutando simulaciones de enrutamiento ====")
    
    # Simulación AcoNet
    print("\nIniciando simulación AcoNet...")
    AcoNet_run = AcoNet(grafo, 
                        no_ants=20, 
                        alpha=0.5, 
                        beta=1.2, 
                        p_factor=10, 
                        no_elite_ants=5, 
                        evaporation_rate=0.005)
    routers_AcoNet = AcoNet_run.simulate_AcoNet()

    print("\nIniciando simulación OSPF...")
    ospf_run = Ospf(grafo)    
    routers_ospf = ospf_run.simulate_ospf()
    
    # Seleccionar origen y destino para las pruebas
    source, destination = select_test_nodes(grafo, seed=433)
    
    # Ejecutar y comparar ambos algoritmos
    if source and destination:
        # Trazar ruta con AcoNet
        path_AcoNet = trace_and_compare(topologia, grafo, source, destination, routers_AcoNet, "AcoNet")
        topologia.plot(path_AcoNet)
        
        # Trazar ruta con OSPF
        path_ospf = trace_and_compare(topologia, grafo, source, destination, routers_ospf, "OSPF")
        topologia.plot(path_ospf)
        
        # Comparación final
        if path_AcoNet and path_ospf:
            AcoNet_length = sum(grafo[path_AcoNet[i]][path_AcoNet[i+1]]['weight'] for i in range(len(path_AcoNet)-1))
            ospf_length = sum(grafo[path_ospf[i]][path_ospf[i+1]]['weight'] for i in range(len(path_ospf)-1))
            
            print("\n==== Comparación de algoritmos ====")
            print(f"Longitud de ruta AcoNet: {AcoNet_length:.2f}")
            print(f"Longitud de ruta OSPF: {ospf_length:.2f}")
            print(f"Diferencia (AcoNet - OSPF): {AcoNet_length - ospf_length:.2f}")
            
            if path_AcoNet == path_ospf:
                print("Ambos algoritmos calcularon la misma ruta.")
            else:
                print("Los algoritmos calcularon rutas diferentes.")

if __name__ == "__main__":
    main()