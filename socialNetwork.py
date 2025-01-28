import networkx as nx
import matplotlib
import random
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SocialNetwork:
    def __init__(self, node_count, recovery_delay, rumor_count=1):
        self.graph = nx.complete_graph(node_count)
        self.node_count = node_count
        self.recovery_delay = recovery_delay
        self.rumor_count = rumor_count
        self.current_step = 0
        self.recovery_started = False

        # Initialize node states: 0 = Susceptible, 1 = Infected, 2 = Recovered
        self.status = {node: 0 for node in self.graph.nodes()}
        self.infection_time = {node: None for node in self.graph.nodes()}

        # Infect one random node
        initial_infected = random.choice(list(self.graph.nodes()))
        self.status[initial_infected] = 1
        self.infection_time[initial_infected] = 0

    def run_time_step(self):
        """
        Execute a single time step of the simulation.
        """
        new_status = self.status.copy()

        # Spread infection
        for node in self.graph.nodes():
            if self.status[node] == 1:  # Infected node
                for neighbor in self.graph.neighbors(node):
                    if self.status[neighbor] == 0 and random.random() < 0.05:  # Infection probability
                        new_status[neighbor] = 1
                        if self.infection_time[neighbor] is None:
                            self.infection_time[neighbor] = self.current_step

        # Check if recovery can start
        if not self.recovery_started and all(s == 1 for s in self.status.values()):
            print("All nodes infected. Recovery now possible.")
            self.recovery_started = True

        # Manage recovery
        for node in self.graph.nodes():
            if self.status[node] == 1 and self.infection_time[node] is not None:
                if self.current_step - self.infection_time[node] >= self.recovery_delay:
                    new_status[node] = 2  # Recovered

        self.status = new_status
        self.current_step += 1

    def visualize(self):
        """
        Save the current state of the social network as an image.
        """
        status_colors = {0: "blue", 1: "red", 2: "green"}
        pos = nx.spring_layout(self.graph, k=2)  # Adjust 'k' for spacing
        node_colors = [status_colors[self.status[node]] for node in self.graph.nodes()]

        output_folder = "SocialNet"
        os.makedirs(output_folder, exist_ok=True)

        save_path = os.path.join(output_folder, f"Rumor{self.rumor_count}_TimeStep{self.current_step}.png")

        plt.figure(figsize=(16, 9))
        plt.clf()
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=50,
            edge_color="gray",
            width=0
        )
        plt.title(f"Social Network - Rumor {self.rumor_count}, Step {self.current_step}")
        plt.savefig(save_path)
        plt.close()

    def is_simulation_complete(self):
        """
        Check if the simulation is complete (no infected nodes left).
        """
        return all(s != 1 for s in self.status.values())


# Main function to interactively run multiple simulations
def main():
    networks = []  # List to hold multiple social networks
    num_networks = 2  # Number of networks to simulate

    # Initialize multiple networks
    for i in range(num_networks):
        networks.append(SocialNetwork(node_count=300, recovery_delay=10, rumor_count=i + 1))

    while True:
        all_complete = True
        for network in networks:
            if not network.is_simulation_complete():
                print(f"Running step {network.current_step} for Rumor {network.rumor_count}")
                network.run_time_step()
                network.visualize()
                all_complete = False

        if all_complete:
            print("All simulations complete.")
            break

        input("Press Enter to proceed to the next step...")


if __name__ == "__main__":
    main()
