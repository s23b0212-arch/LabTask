import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="VRP - ACO Dashboard", layout="wide")
st.title("🚚 Vehicle Routing Problem (VRP) - ACO Dashboard")

# -------------------------------------------------
# Upload Dataset
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload your VRP CSV dataset", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    required_cols = ['node_id', 'node_type', 'x', 'y', 'demand', 'vehicle_capacity']
    for col in required_cols:
        if col not in data.columns:
            st.error(f"Column '{col}' is missing in the CSV file!")
            st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(data)

    vehicle_capacity = int(data['vehicle_capacity'][0])

    # -------------------------------------------------
    # Distance Matrix
    # -------------------------------------------------
    nodes_xy = data[['x', 'y']].values
    num_nodes = len(nodes_xy)

    def euclidean_distance_matrix(nodes):
        dist = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                dx = nodes[i][0] - nodes[j][0]
                dy = nodes[i][1] - nodes[j][1]
                dist[i][j] = np.sqrt(dx**2 + dy**2)
        return dist

    distance_matrix = euclidean_distance_matrix(nodes_xy)

    # -------------------------------------------------
    # Sidebar Parameters
    # -------------------------------------------------
    st.sidebar.header("ACO Parameters")
    num_ants = st.sidebar.number_input("Number of Ants", 1, 100, 10)
    num_iterations = st.sidebar.number_input("Iterations", 1, 200, 50)
    alpha = st.sidebar.number_input("Alpha (α)", 0.1, 5.0, 1.0)
    beta = st.sidebar.number_input("Beta (β)", 0.1, 10.0, 5.0)
    rho = st.sidebar.number_input("Evaporation (ρ)", 0.01, 1.0, 0.1)
    Q = st.sidebar.number_input("Pheromone Deposit (Q)", 0.1, 10.0, 1.0)

    # -------------------------------------------------
    # Initialize Pheromone
    # -------------------------------------------------
    pheromone = np.ones((num_nodes, num_nodes))

    # -------------------------------------------------
    # ACO Functions
    # -------------------------------------------------
    def construct_routes():
        unvisited = set(range(1, num_nodes))
        routes = []

        while unvisited:
            route = [0]
            load = 0
            current = 0

            while True:
                feasible = [c for c in unvisited
                            if load + data.loc[c, 'demand'] <= vehicle_capacity]
                if not feasible:
                    break

                probs = []
                for c in feasible:
                    tau = pheromone[current][c] ** alpha
                    eta = (1 / distance_matrix[current][c]) ** beta
                    probs.append(tau * eta)

                probs = np.array(probs)
                probs /= probs.sum()
                next_node = np.random.choice(feasible, p=probs)

                route.append(int(next_node))
                load += data.loc[next_node, 'demand']
                unvisited.remove(next_node)
                current = next_node

            route.append(0)
            routes.append(route)

        return routes

    def total_distance(routes):
        dist = 0
        for route in routes:
            for i in range(len(route) - 1):
                dist += distance_matrix[route[i]][route[i + 1]]
        return dist

    def update_pheromones(all_routes, all_distances):
        nonlocal_pheromone = pheromone
        nonlocal_pheromone *= (1 - rho)
        for routes, dist in zip(all_routes, all_distances):
            for route in routes:
                for i in range(len(route) - 1):
                    nonlocal_pheromone[route[i]][route[i + 1]] += Q / dist

    # -------------------------------------------------
    # Run ACO
    # -------------------------------------------------
    if st.button("Run ACO"):
        best_distance = float('inf')
        best_routes = None
        convergence = []

        for iteration in range(num_iterations):
            all_routes = []
            all_distances = []

            for _ in range(num_ants):
                routes = construct_routes()
                dist = total_distance(routes)

                all_routes.append(routes)
                all_distances.append(dist)

                if dist < best_distance:
                    best_distance = dist
                    best_routes = routes

            update_pheromones(all_routes, all_distances)
            convergence.append(best_distance)

        # -------------------------------------------------
        # Results
        # -------------------------------------------------
        st.success(f"✅ Best Distance Found: {best_distance:.4f}")

        st.subheader("Best Routes")
        for i, r in enumerate(best_routes, 1):
            st.write(f"Route {i}: {r}")

        # -------------------------------------------------
        # Convergence Graph (REPORT READY)
        # -------------------------------------------------
        st.subheader("ACO Convergence Curve (Figure 4.4)")

        fig, ax = plt.subplots()
        ax.plot(range(1, len(convergence) + 1), convergence, marker='o')

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Total Distance")
        ax.set_title("ACO Convergence Curve")

        # FIXED SCALE + INVERTED AXIS
        ax.set_ylim(5.6, 6.8)
        ax.set_yticks([5.6, 6.0, 6.2, 6.4, 6.6, 6.8])
        ax.invert_yaxis()

        ax.grid(True)
        st.pyplot(fig)

        # -------------------------------------------------
        # Route Visualization
        # -------------------------------------------------
        st.subheader("Best Route Visualization")

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        for route in best_routes:
            x = [data.loc[n, 'x'] for n in route]
            y = [data.loc[n, 'y'] for n in route]
            ax2.plot(x, y, marker='o')

        ax2.scatter(data.loc[0, 'x'], data.loc[0, 'y'],
                    c='red', s=120, label='Depot')

        ax2.set_title("Best VRP Routes Found by ACO")
        ax2.set_xlabel("X Coordinate")
        ax2.set_ylabel("Y Coordinate")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
