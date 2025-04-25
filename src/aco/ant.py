# Clase que representa una hormiga que se mueve por el grafo
# Implementa principalmente la lógica de movimiento de la hormiga y lo que implica
import random

class Ant:
    def __init__(self, starting_node, alpha, beta):
        if starting_node is None:
            self.tabu = []
            self.tour_length = float("inf")
            self.alpha = alpha
            self.beta = beta
        else:
            self.tabu = [starting_node]
            self.current_node = starting_node
            self.tour_length = 0.0
            self.alpha = alpha  # Parámetro para controlar la importancia del rastro
            self.beta = beta  # Parámetro para controlar la importancia de la visibilidad

    def __str__(self):
        if self.tour_length == float("inf"):
            return "[]|inf"
        return self.tabu.__str__() + '|' + self.tour_length.__str__()
    
    def __lt__(self, other):
        return self.tour_length < other.tour_length

    def __eq__(self, other):
        return self.tabu == other.tabu

    def move(self, posciones, edges):
        if len(posciones) == len(self.tabu):
            _, dist = edges[tuple(sorted((self.current_node, self.tabu[0])))] 
            self.tour_length += dist
            self.tabu.append(self.tabu[0])
            self.current_node = self.tabu[0]
            return
        no_visitados = set(list(posciones.keys())) - set(self.tabu)
        probabilidades = {}
        denominador = 0
        for nodo in no_visitados:
            tau_ik, eta_ik = edges[tuple(sorted((self.current_node, nodo)))] 
            eta_ik = 1/eta_ik
            denominador += (tau_ik ** self.alpha) * (eta_ik ** self.beta)
        for next_nodo in no_visitados:
            tau_ij, eta_ij = edges[tuple(sorted((self.current_node, next_nodo)))] 
            eta_ij = 1/eta_ij
            numerador = (tau_ij ** self.alpha) * (eta_ij ** self.beta)  

            probabilidades[next_nodo] = numerador/denominador

        choosen_next_nodo = random.choices(list(probabilidades.keys()), weights=probabilidades.values(), k=1)[0]    
        
        # Activar para hacer greedy el algoritmo
        # choosen_next_nodo = max(probabilidades, key=probabilidades.get)

        _, dist = edges[tuple(sorted((self.current_node, choosen_next_nodo)))] 
        self.tour_length += dist
        self.tabu.append(choosen_next_nodo)
        self.current_node = choosen_next_nodo