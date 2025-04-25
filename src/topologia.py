import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import random
NODE_SIZE = 500
DEFAULT_COLOR = 'lightblue'

class Topologia:
    def __init__(self):
        self.grafo = nx.Graph()
        self.aristas_eliminadas = []

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

    def crear_red(self, no_seg_sec, maximo_nodos_sec, ratio_desviacion=0.05):
        """Crea un grafo de prueba con segmentos primarios y secundarios"""
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
        return G, self.aristas_eliminadas

    def actualizar_pesos(self):
        """Actualiza los pesos de las aristas basándose en distancia y latencia"""
        velocidad_luz = 299_792_458  # m/s
        for u, v, data in self.grafo.edges(data=True):
            distancia = data['distance']  # en metros
            tiempo_transmision = (distancia / velocidad_luz) * 1000  # ms
            latencia_random = random.uniform(10, 1000)  # ms
            self.grafo[u][v]['weight'] = tiempo_transmision + latencia_random

    def trace_route(self, source, destination, routers):
        """Simula traceroute entre source y destination usando tabla de routers"""
        if source not in routers or destination not in routers:
            print("Origen o destino no válido")
            return []
        
        path = [source]
        current = source

        while current != destination:
            next_hop = routers[current].get_next_hop(destination)
            if next_hop is None:
                print(f"No hay ruta desde {source} hasta {destination}")
                return path
            path.append(next_hop)
            current = next_hop
        
        return path

    def plot(self, path=None):
        """Visualiza el grafo, resaltando una ruta si se da"""
        G = self.grafo
        plt.figure(figsize=(12, 8))

        if path and len(path) > 1:
            node_colors = ['red' if node in path else DEFAULT_COLOR for node in G.nodes()]
            path_edges = set((path[i], path[i+1]) for i in range(len(path)-1))

            # Create a new set for reversed edges
            reversed_edges = {(v, u) for u, v in path_edges}
            path_edges.update(reversed_edges)

            edge_colors = []
            widths = []

            for u, v in G.edges():
                if (u, v) in path_edges:
                    edge_colors.append('red')
                    widths.append(2.5)
                else:
                    edge_colors.append('black')
                    widths.append(1.0)
            nx.draw(G, with_labels=True, node_size=NODE_SIZE, node_color=node_colors,
                    font_size=10, font_weight='bold', edge_color=edge_colors, width=widths)
        else:
            nx.draw(G, with_labels=True, node_size=NODE_SIZE, node_color=DEFAULT_COLOR,
                    font_size=10, font_weight='bold')

        plt.title("Grafo de red" + (" con ruta resaltada" if path else ""))
        plt.show()
