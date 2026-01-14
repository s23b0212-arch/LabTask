import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Page configuration
# --------------------------
st.set_page_config(page_title="VRP using Ant Colony Optimization", layout="wide")
st.title("🚚 Vehicle Routing Problem (VRP) using Ant Colony Optimization")

st.markdown("""
This application solves the Vehicle Routing Problem (VRP) using an evolutionary
computation approach based on **Ant Colony Optimization (ACO)**.  
It supports **single-objective** and **multi-objective** optimization.
""")

# --------------------------
# Dataset Upload
# --------------------------
uploaded_file = st.file_uploader("Upload VRP Dataset (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    required_cols = ["x", "y", "demand"]
    for col in required_cols:
        if col not in data.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    st.subheader("🖼️ Figure 1: Dataset Preview")
    st.dataframe(data)

    vehicle_capacity = int(data["vehicle_capacity"][0])

    # --------------------------
    # Distance Matrix
    # --------------------------
    coords = data[["x", "y"]].values
    n_nodes = len(coords)

    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    # --------------------------
    # Sidebar Controls
    # --------------------------
    st.sidebar.header("🖼️ Figure 2: ACO Parameters")

    num_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
    iterations = st.sidebar.slider("Iterations", 10, 200, 80)
    alpha = st.sidebar.slider("Alpha (Pheromone)", 0.1, 5.0, 1.0)
    beta = st.sidebar.slider("Beta (Heuristic)", 1.0, 10.0, 5.0)
    rho = st.sidebar.slider("Evaporation Rate (ρ)", 0.01, 0.9, 0.1)
    Q = st.sidebar.slider("Pheromone Deposit (Q)", 0.1, 10.0, 1.0)

    objective_mode = st.sidebar.selectbox(
        "Optimization Mode",
        ["Single Objective (Distance)", "Multi Objective (Distance + Vehicles)"]
    )

    # --------------------------
    # ACO Functions
    # --------------------------
    pheromone = np.ones((n_nodes, n_nodes))

    def construct_routes():
        unvisited = set(range(1, n_nodes))
        routes = []

        while unvisited:
            route = [0]
            load = 0
            current = 0

            while True:
                feasible = [
                    c for c in unvisited
                    if load + data.loc[c, "demand"] <= vehicle_capacity
                ]
                if not feasible:
                    break

                probs = []
                for c in feasible:
                    tau = pheromone[current][c] ** alpha
                    eta = (1 / (dist_matrix[current][c] + 1e-6)) ** beta
                    probs.append(tau * eta)

                probs = np.array(probs)
                probs /= probs.sum()

                next_node = np.random.choice(feasible, p=probs)
                route.append(next_node)
                load += data.loc[next_node, "demand"]
                unvisited.remove(next_node)
                current = next_node

            route.append(0)
            routes.append(route)

        return routes

    def total_distance(routes):
        d = 0
        for r in routes:
            for i in range(len(r) - 1):
                d += dist_matrix[r[i]][r[i + 1]]
        return d

    def fitness(routes):
        distance = total_distance(routes)
        vehicles = len(routes)

        if objective_mode == "Multi Objective (Distance + Vehicles)":
            return distance + 10 * vehicles
        return distance

    def update_pheromone(all_routes, all_scores):
        pheromone[:] *= (1 - rho)
        for routes, score in zip(all_routes, all_scores):
            for r in routes:
                for i in range(len(r) - 1):
                    pheromone[r[i]][r[i + 1]] += Q / score

    # --------------------------
    # Run Algorithm
    # --------------------------
    if st.button("🚀 Run ACO"):
        best_score = float("inf")
        best_routes = None
        convergence = []

        for it in range(iterations):
            routes_list = []
            scores = []

            for _ in range(num_ants):
                routes = construct_routes()
                score = fitness(routes)

                routes_list.append(routes)
                scores.append(score)

                if score < best_score:
                    best_score = score
                    best_routes = routes

            update_pheromone(routes_list, scores)
            convergence.append(best_score)

        # --------------------------
        # Results
        # --------------------------
        st.subheader("🖼️ Figure 3: Best Total Distance")
        st.success(f"Best Fitness Value: {best_score:.2f}")

        st.subheader("🖼️ Figure 4: Best Routes Found")
        for i, r in enumerate(best_routes, 1):
            st.write(f"Route {i}: {r}")

        st.subheader("🖼️ Figure 5: Convergence Curve")
        fig1, ax1 = plt.subplots()
        ax1.plot(convergence)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Fitness")
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("🖼️ Figure 6: Route Visualization")
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        for r in best_routes:
            x = [coords[n][0] for n in r]
            y = [coords[n][1] for n in r]
            ax2.plot(x, y, marker="o")

        ax2.scatter(coords[0][0], coords[0][1], c="red", s=120, label="Depot")
        ax2.set_title("Optimized VRP Routes")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
