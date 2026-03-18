# Optimization of the Traveling Salesman Problem (TSP) via Simulated Annealing

[cite_start]University project for the *Optimization and Statistical Mechanics* exam (Physics of Complex Systems and Big Data)[cite: 139, 140].

## 🎯 Project Objective
[cite_start]Given $N$ nodes and a distance matrix, the TSP requires finding the shortest closed path to visit all of them exactly once[cite: 154]. [cite_start]Since the number of possible paths scales as $(N-1)!$, this is a complex combinatorial optimization problem[cite: 155, 156, 157]. 
[cite_start]This project leverages Simulated Annealing to slowly "cool" the system toward its energetic *ground state* (minimum cost), avoiding getting trapped in metastable states[cite: 161].

## 🧠 How the Simulated Annealing Algorithm Works
The Simulated Annealing (S.A.) algorithm used in this project is structured in consecutive cycles of **Cooling** and **Heating** to find the optimal path. Here is the step-by-step evolution:

1. **Initial Setup:** We start with a random path (tour) visiting all $N$ nodes. We set an initial high temperature $T_{max}$ (which corresponds to a very low $\beta$, since $T \propto 1/\beta$). This allows the system to explore the solution space freely.
2. **State Perturbation:** At each step, we generate a new candidate tour $Y$ by slightly mutating the current tour $X$ (e.g., swapping the visit order of two random cities).
3. **Metropolis Criterion:** We calculate the cost difference $\Delta C = C(Y) - C(X)$. 
   * If $\Delta C < 0$, the new tour is better and is always accepted.
   * If $\Delta C > 0$, the new tour is worse, but it might still be accepted with a probability $p = e^{-\beta \Delta C}$.
4. **Cooling Phase:** We gradually increase $\beta$ (decreasing the temperature). As the system cools, the probability of accepting worse solutions drops exponentially. This forces the algorithm to settle into a local or global minimum (fine-tuning).
5. **Heating Phase (Reheating):** Once the minimum temperature $T_{min}$ is reached, the system is slowly heated back up (decreasing $\beta$). This gives the algorithm a burst of energy to "climb out" of any suboptimal local minima it might have fallen into.
6. **Annealing Cycles:** Steps 4 and 5 are repeated for a set number of `annealing_cycles`. The absolute best tour found across all cycles is then returned as the final solution.

*Below, the static and dynamic evolution of Cost and Temperature over the annealing cycles, alongside the physical routing optimization on a synthetic modular graph:*

![Heating and Cooling Cycles](assets/tsp_sa_4cicli_1run_N30.jpg)
![Heating and Cooling GIF](assets/tsp_sa_costi_temperatura.gif)
![Graph Evolution GIF](assets/tsp_sa_singola_run_grafo.gif)

## 📊 The Three Main Studies
[cite_start]The project is structured into three analytical phases using synthetic datasets and one on real geographic data[cite: 142, 146, 148].

### 1. Convergence to the Ground State
[cite_start]For small numbers of nodes ($N < 12$), we verified that the method has a high probability of converging to the exact ground state rather than a metastable state[cite: 202]. [cite_start]The probability is maximum (100%) for $N < 9$ and remains above 60% up to $N = 11$[cite: 202]. [cite_start]For larger N, this probability is expected to tend to 0[cite: 203]. [cite_start]Execution time grows very slowly compared to the exact algorithm, which explodes factorially[cite: 205].

![Probability of Ground State vs N](assets/grafico_tsp_P_vs_N_sintetico_pp1.png)

### 2. Method Stability
[cite_start]At a fixed N, the method is stable regardless of the distance matrix or the initial tour[cite: 201]. [cite_start]The relative percentage error on the minimum cost value remains extremely low, peaking at merely $\approx 0.35\%$ around $N=48$[cite: 201].

![Method Stability](assets/tsp_grafico_stabilità_pp_normalizzato_mediato.png)

### 3. Optimal Temperature Range ($T_{max}$)
[cite_start]Using the optimal initial temperature, the iterative S.A. method can provide satisfactory results even for very large real-world problems[cite: 204]. We tested a wide range of temperatures to locate the absolute minimum cost zone, effectively avoiding both under-exploration (low T) and computational waste (excessively high T).

![Optimal Temperature Range](assets/tsp_sa_ww_rangeT_15run_minimo_abs.png)

## 🌍 Real-World Application (GeoPandas)
Finally, the algorithm was applied to real geographic coordinates. [cite_start]Below is the routing optimization applied to 45 European capitals[cite: 274], dynamically minimizing the travel distance.

![European Capitals Routing](assets/tsp_capitali_europee4.gif)

## 🛠️ Tech Stack & Requirements
* **Language:** Python
* **Main Libraries:** `NumPy`, `SciPy`, `NetworkX`, `Matplotlib`, `GeoPandas`, `Contextily`, `adjustText`.
To run the geographic scripts, ensure you have the required spatial libraries installed:
`pip install geopandas contextily adjustText shapely`