import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="VRP - ACO Dashboard", layout="wide")
st.title("🚚 Vehicle Routing Problem (VRP) using Ant Colony Optimization")

# --------------------------
# Upload Dataset
# --------------------------
uploaded_file = st.file_uploader("Upload VRP CSV Dataset", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Required columns check
    required_cols = ["x", "y", "demand"]
    for col in required_cols:
        if col not in data.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # --------------------------
    # Figure 1: Dataset Preview
    # --------------------------
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
            dist_matrix[i][j] = np.sqrt(
                (coords[i][0] - coords[j][0])**2 +
                (coords[i][1] - coords[j][1])**2
            )

    # --------------------------
    # Figure 2: ACO Parameters
    # --------------------------
    st.sidebar.header("🖼️ Figure 2: ACO Parameter Control Panel")

    num_ants = st.sidebar.number_input("Number of Ants", 1, 50, 10)
    iterations = st.sidebar.number_input("Iterations", 1, 200, 50)
    alpha = st.sidebar.number_input("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
    beta = st.sidebar.number_input("Beta (heuristic importance)", 0.1, 10.0, 5.0)
    rho = st.sidebar.number_input("Rho (evaporation rate)", 0.01, 1.0, 0.1)
    Q = st.sidebar.number_input("Q (pheromone deposit)", 0.1, 10.0, 1.0)

    pheromone = np.ones((n_nodes, n_nodes))

    # --------------------------
    # ACO Functions
    # --------------------------
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
                    eta = (1 / dist_matrix[current][c]) ** beta
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

    def update_pheromone(all_routes, all_distances):
        pheromone[:] *= (1 - rho)
        for routes, dist in zip(all_routes, all_distances):
            for r in routes:
                for i in range(len(r) - 1):
                    pheromone[r[i]][r[i + 1]] += Q / dist

    # --------------------------
    # Run ACO
    # --------------------------
    if st.button("Run ACO"):
        best_distance = float("inf")
        best_routes = None
        convergence = []

        for _ in range(iterations):
            all_routes = []
            all_distances = []

            for _ in range(num_ants):
