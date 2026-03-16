# optimization-traveling-salseman-problem
university project

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