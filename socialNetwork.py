import networkx as nx
import matplotlib
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Function to initialize the social network with SIR model
def initialize_social_network_sir(node_count):
    g = nx.complete_graph(node_count)

    # SIR Model configuration
    model = ep.SIRModel(g)
    config = mc.Configuration()
    config.add_model_parameter('beta', 0.05)  # Infection rate
    config.add_model_parameter('gamma', 0.01)  # Recovery rate
    model.set_initial_status(config)

    # Set one initial infected node
    initial_infected = random.choice(list(g.nodes()))
    config.add_model_initial_configuration('Infected', [initial_infected])
    model.set_initial_status(config)

    return model, g


# Visualize the social network and SIR model propagation
def visualize_sir_propagation(node_count, model, graph, steps=50, delay=1):
    status_colors = {"Susceptible": "blue", "Infected": "red", "Removed": "green"}
    pos = nx.spring_layout(graph)  # Position for visualization
    plt.ion()  # Interactive mode

    for step in range(steps):
        # Execute one iteration of the model
        iterations = model.iteration()
        status = iterations['status']

        # Ensure all nodes are accounted for in the status dictionary
        status = {node: status.get(node, 0) for node in graph.nodes()}

        # Determine node colors based on status
        node_colors = [
            status_colors["Susceptible"] if status[node] == 0 else
            status_colors["Infected"] if status[node] == 1 else
            status_colors["Removed"]
            for node in graph.nodes()
        ]

        # Visualize the graph
        plt.clf()
        nx.draw(
            graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=50,
            edge_color="gray"
        )
        plt.title(f"SIR Model - Step {step + 1}")
        plt.draw()
        plt.pause(delay)  # Pause for visualization (adjust delay for speed)

        # Break if no more infected nodes
        if all(s != 1 for s in status.values()):
            break

    plt.ioff()
    plt.show()


# Main function to test the SIR model implementation
def main():
    node_count = 100  # Number of nodes
    social_model, social_graph = initialize_social_network_sir(node_count)

    # Visualize SIR model propagation
    visualize_sir_propagation(node_count, social_model, social_graph, steps=50, delay=0.5)


if __name__ == "__main__":
    main()