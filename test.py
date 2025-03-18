import andes
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing

def create_and_visualize_network():
    # Initialize the ANDES system
    system = andes.System()

    # Add buses using the generic 'add' method
    system.add("Bus", idx=1, name='Bus 1', voltage=1.0)
    system.add("Bus", idx=2, name='Bus 2', voltage=1.0)
    system.add("Bus", idx=3, name='Bus 3', voltage=1.0)

    # Add branches (transmission lines) connecting the buses
    system.add("Branch", idx=1, fbus=1, tbus=2, r=0.01, x=0.1, b=0.01)
    system.add("Branch", idx=2, fbus=2, tbus=3, r=0.01, x=0.1, b=0.01)
    system.add("Branch", idx=3, fbus=3, tbus=1, r=0.01, x=0.1, b=0.01)

    # Prepare the system (this may trigger lazy initialization and processing)
    system.prepare(quick=True)

    # Create a NetworkX graph for visualization
    G = nx.Graph()

    # Retrieve bus components from the system
    buses = system.get_components("Bus")
    for bus in buses:
        # Each bus is assumed to have attributes 'idx' and 'name'
        G.add_node(bus.idx, label=bus.name)

    # Retrieve branch components from the system
    branches = system.get_components("Branch")
    for branch in branches:
        # Create an edge between the from bus (fbus) and the to bus (tbus)
        G.add_edge(branch.fbus, branch.tbus)

    # Generate a layout and draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
    plt.title("Simple ANDES Power Network")
    plt.show()

if __name__ == '__main__':
    # Ensure proper multiprocessing support on Windows.
    multiprocessing.freeze_support()
    create_and_visualize_network()
