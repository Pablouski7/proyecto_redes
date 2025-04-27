import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import random
NODE_SIZE = 500
DEFAULT_COLOR = 'lightblue'

class Topologia:
    def __init__(self, seed=None):
        self.grafo = nx.Graph()
        self.aristas_eliminadas = []
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calcula distancia esférica usando fórmula de Haversine"""
        R = 6371  # Radio de la Tierra en km
        phi1, phi2 = radians(lat1), radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)
        a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        d = R * c
        return d

    def crear_red(self, no_seg_sec, maximo_nodos_sec, ratio_desviacion=0.05, seed=None):
        """Crea un grafo de prueba con segmentos primarios y secundarios
        
        Args:
            no_seg_sec: Número de segmentos secundarios
            maximo_nodos_sec: Máximo número de nodos secundarios por segmento
            ratio_desviacion: Ratio de desviación para la ubicación de nodos secundarios
            seed: Semilla para controlar la reproducibilidad (None usa la semilla global)
        """
        # Establecer la semilla si se proporciona una nueva
        if seed is not None:
            old_state = random.getstate()
            random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            old_state = random.getstate()
            random.seed(self.seed)

        G = self.grafo

        # Crear nodos primarios
        for i in range(no_seg_sec):
            G.add_node(f"P{i}", lat=random.uniform(-90, 90), lon=random.uniform(-180, 180))
        
        # Crear nodos secundarios
        for i in range(no_seg_sec):
            num_nodos_sec = random.randint(1, maximo_nodos_sec)
            for j in range(num_nodos_sec):
                max_lat = G.nodes[f"P{i}"]['lat'] + ratio_desviacion * random.uniform(-1, 1)
                min_lat = G.nodes[f"P{i}"]['lat'] - ratio_desviacion * random.uniform(-1, 1)
                max_lon = G.nodes[f"P{i}"]['lon'] + ratio_desviacion * random.uniform(-1, 1)
                min_lon = G.nodes[f"P{i}"]['lon'] - ratio_desviacion * random.uniform(-1, 1)
                lat = random.uniform(min_lat, max_lat)
                lon = random.uniform(min_lon, max_lon)
                nodo_sec = f"SS{i}_{j}"
                G.add_node(nodo_sec, lat=lat, lon=lon)
                G.add_edge(nodo_sec, f"P{i}")

        # Conectar primarios completamente
        for i in range(no_seg_sec):
            for j in range(i + 1, no_seg_sec):
                G.add_edge(f"P{i}", f"P{j}")

        # Eliminar aleatoriamente algunas conexiones entre primarios
        aristas = list(G.edges())
        num_eliminar = random.randint(1, len(aristas) // 2)
        for _ in range(num_eliminar):
            arista = random.choice(aristas)
            if arista[0][0] != "P" or arista[1][0] != "P":
                continue
            if (arista not in self.aristas_eliminadas) and ((arista[1], arista[0]) not in self.aristas_eliminadas):
                G.remove_edge(*arista)
                self.aristas_eliminadas.append(arista)

        # Calcular distancias Haversine
        for u, v in G.edges():
            lat1, lon1 = G.nodes[u]['lat'], G.nodes[u]['lon']
            lat2, lon2 = G.nodes[v]['lat'], G.nodes[v]['lon']
            distancia = self.haversine(lat1, lon1, lat2, lon2)
            G[u][v]['distance'] = distancia * 1000  # metros
        
        self.grafo = G
        self.actualizar_pesos()  # Actualizar pesos después de crear el grafo

        # Restaurar el estado aleatorio anterior si se cambió
        if seed is not None or self.seed is not None:
            random.setstate(old_state)
            
        return G, self.aristas_eliminadas

    def actualizar_pesos(self, seed=None):
        """Actualiza los pesos de las aristas basándose en distancia y latencia
        
        Args:
            seed: Semilla para controlar la reproducibilidad de las latencias
        """
        # Establecer la semilla si se proporciona una nueva
        if seed is not None:
            old_state = random.getstate()
            random.seed(seed)
        elif self.seed is not None:
            old_state = random.getstate()
            random.seed(self.seed)
            
        velocidad_luz = 299_792_458  # m/s
        for u, v, data in self.grafo.edges(data=True):
            distancia = data['distance']  # en metros
            tiempo_transmision = (distancia / velocidad_luz) * 1000  # ms
            latencia_random = random.uniform(10, 1000)  # ms
            self.grafo[u][v]['weight'] = tiempo_transmision + latencia_random
            
        # Restaurar el estado aleatorio anterior si se cambió
        if seed is not None or self.seed is not None:
            random.setstate(old_state)

    def trace_route(self, source, destination, routers):
        """Simula traceroute entre source y destination usando tabla de routers"""
        if source not in routers or destination not in routers:
            print("Origen o destino no válido")
            return []
        visited = set()
        path = [source]
        current = source
        probados = set()
        while current != destination:
            next_hop = routers[current].get_next_hop(destination, visited, probados)
            routers[current].print_routing_table()
            if next_hop is None and source not in probados:

                print("Añadiendo nodo a probados:", current)
                print("path antes de pop:", path)   
                print("visited antes de pop:", visited)
                probados.add(current)
                visited.remove(current)
                path.pop()
                current = path[-1] 
                
                print("path después de pop:", path)
                print("visited después de pop:", visited)
                continue
            else:
                visited.add(current)
                if next_hop is None:
                    print(f"No hay ruta desde {source} hasta {destination}")
                    return path
            
            path.append(next_hop)
            current = next_hop
        
        return path

    def plot(self, path=None):
        """Visualiza el grafo de forma estética, resaltando rutas si se dan."""
        G = self.grafo
        plt.figure(figsize=(16, 10))  # Más grande para que no se vea apretado
        
        # Usar spring_layout para más separación
        pos = nx.spring_layout(G, k=0.1, iterations=50, seed=self.seed)
        
        # Colores
        P_COLOR = '#00b4d8'  # Azul celeste para primarios
        SS_COLOR = 'purple'  # Amarillo para secundarios
        PATH_COLOR = 'red'  # Rojo para resaltar rutas
        
        # Tamaños
        PRIMARY_SIZE = 700
        SECONDARY_SIZE = 300
        
        # Clasificar nodos
        primary_nodes = [n for n in G.nodes if n.startswith('P')]
        secondary_nodes = [n for n in G.nodes if n.startswith('SS')]

        # Preparar nodos en la ruta si hay
        path_nodes = set(path) if path else set()
        path_edges = set()
        if path and len(path) > 1:
            path_edges = {(path[i], path[i+1]) for i in range(len(path)-1)}
            path_edges |= {(path[i+1], path[i]) for i in range(len(path)-1)}
        
        # Dibujar nodos primarios
        nx.draw_networkx_nodes(G, pos,
                            nodelist=primary_nodes,
                            node_color=P_COLOR,
                            node_size=PRIMARY_SIZE,
                            )
        
        # Dibujar nodos secundarios
        nx.draw_networkx_nodes(G, pos,
                            nodelist=secondary_nodes,
                            node_color=SS_COLOR,
                            node_size=SECONDARY_SIZE,
                            )
        
        # Dibujar nodos de la ruta en diferente color
        if path_nodes:
            nx.draw_networkx_nodes(G, pos,
                                nodelist=path_nodes,
                                node_color=PATH_COLOR,
                                node_size=PRIMARY_SIZE,
                                )
        
        # Dibujar aristas
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if (u, v) in path_edges or (v, u) in path_edges:
                edge_colors.append(PATH_COLOR)
                edge_widths.append(3)
            elif u.startswith('P') and v.startswith('P'):
                edge_colors.append('forestgreen')
                edge_widths.append(2)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)
        
        # Etiquetas
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')

        plt.axis('off')
        plt.tight_layout()
        plt.show()
