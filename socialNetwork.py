import networkx as nx

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import time


# Function to initialize the social network
def initialize_social_network(car_total):
    g = nx.complete_graph(car_total)
    # Algorithmic Bias model
    model = op.AlgorithmicBiasModel(g)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", 0.32)
    config.add_model_parameter("gamma", 0)
    model.set_initial_status(config)

    return model, g


# Function to propagate the rumor
def propagate_rumor(model, steps=1):
    # Simulate rumor propagation for the given number of steps
    iterations = model.iteration_bunch(steps)
    statuses = iterations[-1]["status"]  # Get the latest opinions
    return statuses


# Real-Time Visualization of Rumor Propagation
def visualize_rumor_propagation(car_total, model, graph, steps=100, delay=1):
    # Create a plot for the network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)  # Fixed layout for consistent visualization

    for step in range(steps):
        # Propagate the rumor for one step
        statuses = propagate_rumor(model, steps=1)
        print(statuses)

        # Update node colors based on their belief in the rumor
        node_colors = [
            "red" if statuses.get(node, 0) >= 0.5 else "blue"
            for node in graph.nodes()
        ]

        # Clear and redraw the graph with updated colors
        plt.clf()
        nx.draw(
            graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            edge_color="gray"
        )
        plt.title(f"Social Network - Step {step + 1}")
        plt.pause(delay)  # Pause for visualization (adjust `delay` for speed)

    plt.show()


# Example Usage
if __name__ == "__main__":
    car_total = 3000  # Number of nodes (cars)
    social_model, social_graph = initialize_social_network(car_total)

    # Visualize the network with rumor propagation in real-time
    visualize_rumor_propagation(car_total, social_model, social_graph, steps=100, delay=1)
