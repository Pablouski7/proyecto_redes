# Deber 2 Inteligencia Artficicial
# Autor: Pablo Herrera
# Fecha: 05/11/2024
# Ejercicio 3: Implementar el algoritmo ACO para resolver TSP.

from aco.aco_as import ACO_AS
import concurrent.futures

def run_aco(N, max_epochs, alpha, beta, rho, Q, nxn):
    aco = ACO_AS(num_ants = N, num_iterations=max_epochs, alpha = alpha, beta = beta, rho = rho, Q = Q, nxn=nxn)
    aco.run()

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            # executor.submit(run_aco, 5*5, 100, 1, 2, 0.5, 100, True),
            # executor.submit(run_aco, 100, 10, 1, 5, 0.5, 100, True),
            # executor.submit(run_aco, 15*15, 30, 0.5, 5, 0.5, 100, True),
            executor.submit(run_aco, 25, 20, 1, 5, 0.5, 100, False),
            executor.submit(run_aco, 100, 20, 0.5, 5, 0.5, 100, False),
            executor.submit(run_aco, 225, 20, 0.5, 5, 0.5, 100, False),
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
