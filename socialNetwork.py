import networkx as nx
import matplotlib
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



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


# Function to propagate rumor with decay
def propagate_rumor_with_decay(model, statuses, step, decay_steps=30):
    new_statuses = {}
    for node, belief in statuses.items():
        if belief > 0:  # Node believes the rumor
            # Apply decay
            new_belief = max(0, belief - (1 / decay_steps))
            new_statuses[node] = new_belief
        else:
            # Node does not believe the rumor, random chance to hear it from neighbors
            neighbors = list(model.graph.neighbors(node))
            if any(statuses.get(neighbor, 0) > 0 for neighbor in neighbors):
                new_statuses[node] = random.uniform(0.5, 1.0)  # Hear the rumor
            else:
                new_statuses[node] = 0
    return new_statuses


# Visualize the social network and rumor propagation
def visualize_rumor_propagation(car_total, model, graph, decay_steps=30, steps=50, delay=0.5):
    statuses = {node: 0 for node in graph.nodes()}  # Initialize all nodes to not believe the rumor
    # Introduce the rumor at a random node
    initial_node = random.choice(list(graph.nodes()))
    statuses[initial_node] = 1.0  # Full belief for the initial node

    pos = nx.spring_layout(graph)  # Position for visualization
    plt.ion()  # Interactive mode

    for step in range(steps):
        # Update statuses with decay
        statuses = propagate_rumor_with_decay(model, statuses, step, decay_steps)

        # Visualize the graph
        node_colors = ["red" if statuses[node] > 0 else "blue" for node in graph.nodes()]
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
        plt.draw()
        plt.pause(delay)  # Pause for visualization (adjust delay for speed)

    plt.ioff()
    plt.show()


# Main function to test the implementation
def main():
    car_total = 10  # Number of nodes (cars)
    social_model, social_graph = initialize_social_network(car_total)

    # Visualize rumor propagation with decay
    visualize_rumor_propagation(car_total, social_model, social_graph, decay_steps=30, steps=50, delay=0.5)


if __name__ == "__main__":
    main()
