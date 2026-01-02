import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random

st.set_page_config(page_title="VRP - ACO Dashboard", layout="wide")

st.title("🚚 Vehicle Routing Problem (VRP) - ACO Dashboard")

# --------------------------
# Step 1: Upload dataset
# --------------------------
uploaded_file = st.file_uploader("Upload your VRP CSV dataset", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # --------------------------
    # Step 2: Extract coordinates and distance matrix
    # --------------------------
    nodes_xy = data[['x','y']].values

    def euclidean_distance_matrix(nodes_xy):
        num_nodes = len(nodes_xy)
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                dx = nodes_xy[i][0] - nodes_xy[j][0]
                dy = nodes_xy[i][1] - nodes_xy[j][1]
                dist_matrix[i][j] = np.sqrt(dx**2 + dy**2)
        return dist_matrix

    distance_matrix = euclidean_distance_matrix(nodes_xy)

    # --------------------------
    # Step 3: ACO Parameters (Interactive)
    # --------------------------
    st.sidebar.header("ACO Parameters")
    num_ants = st.sidebar.number_input("Number of Ants", min_value=1, value=10)
    num_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, value=50)
    alpha = st.sidebar.number_input("Alpha (pheromone importance)", min_value=0.1, value=1.0)
    beta = st.sidebar.number_input("Beta (heuristic importance)", min_value=0.1, value=5.0)
    rho = st.sidebar.number_input("Rho (pheromone evaporation)", min_value=0.01, max_value=1.0, value=0.1)
    Q = st.sidebar.number_input("Q (pheromone deposit)", min_value=0.1, value=1.0)
    vehicle_capacity = st.sidebar.number_input("Vehicle Capacity", min_value=1, value=int(data['vehicle_capacity'][0]))

    # --------------------------
    # Step 4: Initialize pheromone
    # --------------------------
    num_nodes = len(data)
    pheromone = np.ones((num_nodes, num_nodes))

    # --------------------------
    # Step 5: Construct routes
    # --------------------------
    def construct_routes(distance_matrix, pheromone, alpha, beta, vehicle_capacity):
        num_nodes = len(distance_matrix)
        all_customers = set(range(1, num_nodes))
        unvisited = all_customers.copy()
        routes = []

        while unvisited:
            route = [0]
            load = 0
            current_node = 0

            while True:
                feasible_customers = [c for c in unvisited if load + data.loc[c, 'demand'] <= vehicle_capacity]
                if not feasible_customers:
                    break

                probs = []
                for c in feasible_customers:
                    tau = pheromone[current_node][c] ** alpha
                    eta = (1.0 / distance_matrix[current_node][c]) ** beta
                    probs.append(tau * eta)
                probs = np.array(probs)
                probs = probs / probs.sum()
                next_customer = np.random.choice(feasible_customers, p=probs)

                route.append(next_customer)
                load += data.loc[next_customer, 'demand']
                unvisited.remove(next_customer)
                current_node = next_customer

            route.append(0)
            routes.append(route)

        return routes

    # Total distance function
    def total_distance(routes, distance_matrix):
        distance = 0
        for route in routes:
            for i in range(len(route) - 1):
                distance += distance_matrix[route[i]][route[i + 1]]
        return distance

    # Update pheromone
    def update_pheromones(pheromone, all_routes, all_distances, rho=0.1, Q=1.0):
        pheromone *= (1 - rho)
        for routes, dist in zip(all_routes, all_distances):
            for route in routes:
                for i in range(len(route)-1):
                    pheromone[route[i]][route[i+1]] += Q / dist

    # --------------------------
    # Step 6: Run ACO main loop
    # --------------------------
    if st.button("Run ACO"):
        best_distance = float('inf')
        best_routes = None
        convergence = []

        for iteration in range(num_iterations):
            all_routes = []
            all_distances = []

            for ant in range(num_ants):
                routes = construct_routes(distance_matrix, pheromone, alpha, beta, vehicle_capacity)
                dist = total_distance(routes, distance_matrix)
                all_routes.append(routes)
                all_distances.append(dist)

                if dist < best_distance:
                    best_distance = dist
                    best_routes = routes

            update_pheromones(pheromone, all_routes, all_distances, rho, Q)
            convergence.append(best_distance)

        st.success(f"✅ ACO Completed! Best Total Distance: {best_distance:.4f}")

        # --------------------------
        # Step 7: Display Best Routes
        # --------------------------
        st.subheader("Best Routes Found")
        for i, route in enumerate(best_routes, 1):
            st.write(f"Route {i}: {route}")

        # Plot convergence
        st.subheader("Convergence Over Iterations")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(convergence)+1), convergence, marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Total Distance")
        ax.set_title("ACO Convergence Curve")
        st.pyplot(fig)

        # Plot routes
        st.subheader("Route Visualization")
        fig, ax = plt.subplots(figsize=(8,6))
        for route in best_routes:
            x = [data.loc[node, 'x'] for node in route]
            y = [data.loc[node, 'y'] for node in route]
            ax.plot(x, y, marker='o', linestyle='-', alpha=0.7)
        ax.scatter(data.loc[0, 'x'], data.loc[0, 'y'], c='red', s=100, label='Depot')
        ax.set_title("Best VRP Routes Found by ACO")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
