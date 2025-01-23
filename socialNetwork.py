import networkx as nx
import matplotlib
import random

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Function to initialize the social network with custom SIR model
def initialize_social_network_sir(node_count):
    g = nx.complete_graph(node_count)

    # Initialize node states: 0 = Susceptible, 1 = Infected, 2 = Recovered
    status = {node: 0 for node in g.nodes()}
    infection_time = {node: None for node in g.nodes()}  # Track infection times
    initial_infected = random.choice(list(g.nodes()))
    status[initial_infected] = 1  # Start with one infected node
    infection_time[initial_infected] = 0  # Infection starts at step 0

    return g, status, infection_time


# Function to execute a single time step
def single_time_step(graph, status, infection_time, current_step, recovery_delay, recovery_started):
    new_status = status.copy()

    # Infection spread
    for node in graph.nodes():
        if status[node] == 1:  # Infected node
            for neighbor in graph.neighbors(node):
                if status[neighbor] == 0 and random.random() < 0.05:  # Infection probability
                    new_status[neighbor] = 1
                    if infection_time[neighbor] is None:
                        infection_time[neighbor] = current_step  # Record infection time

    # Check if recovery can start
    if not recovery_started and all(s == 1 for s in status.values()):
        print("All nodes infected. Recovery now possible.")
        recovery_started = True

    # Recovery management
    for node in graph.nodes():
        if status[node] == 1 and infection_time[node] is not None:
            # Check if the recovery delay has passed since infection
            if current_step - infection_time[node] >= recovery_delay:
                new_status[node] = 2  # Recovered

    return new_status, recovery_started


# Visualization function
def visualize_sir(graph, status):
    status_colors = {0: "blue", 1: "red", 2: "green"}  # Susceptible, Infected, Removed
    pos = nx.spring_layout(graph)  # Position for visualization
    node_colors = [status_colors[status[node]] for node in graph.nodes()]

    plt.clf()
    nx.draw(
        graph,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=50,
        edge_color="gray"
    )
    plt.title("SIR Model - Current Step")
    plt.draw()
    plt.pause(0.1)  # Pause for visualization


# Main function to interactively run the simulation
def main():
    node_count = 100  # Number of nodes
    recovery_delay = 10  # Time steps before recovery after infection
    social_graph, node_status, infection_time = initialize_social_network_sir(node_count)

    recovery_started = False
    current_step = 0

    # Visualization setup
    plt.ion()

    while True:
        print(f"Step: {current_step}")
        visualize_sir(social_graph, node_status)

        # Execute a single time step
        node_status, recovery_started = single_time_step(
            social_graph, node_status, infection_time, current_step, recovery_delay, recovery_started
        )

        # Check if the simulation is complete
        if all(s != 1 for s in node_status.values()):
            print("Simulation complete. No more infected nodes.")
            break

        current_step += 1
        input("Press Enter to proceed to the next step...")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
