import os
import random
import time
import copy
import csv
import pypsa
import matplotlib.pyplot as plt
import networkx as nx

from powerNetworkGen import (
    get_traffic_lights_from_sumo,
    get_road_edges_from_sumo,
    create_power_network,
    build_networkx_graph,
    get_powered_nodes,
    set_node_down,
    set_node_up
)

os.environ["PROJ_LIB"] = r"C:\\Users\\kth258\\AppData\\Local\\anaconda3\\envs\\sot\\Library\\share\\proj"

############################
# POWER FLOW
############################
def run_power_flow(network):
    """
    Attempts a power flow with Q-limits disabled, then a normal .pf() call
    without initial_solution (for older PyPSA versions).
    """
    if network.generators.empty:
        print("⚠️ No generator. Skipping power flow.")
        return

    try:
        # 1) Turn off Q-limits to avoid reactive constraints
        network.enforce_Q_limits = False

        # 2) Attempt a linear load flow warm-start
        network.lpf()

        # 3) Nonlinear PF but remove 'initial_solution' argument for older versions
        network.pf()
        # If partial islands or lines are too small, you might still see warnings.

    except Exception as e:
        print(f"⚠️ PF failed: {e}")

############################
# VISUALIZATION
############################
def visualize_network_state(network, down_nodes, time_step=0):
    """
    Plot the network with color-coded states:
      Red    = node is 'down'
      Green  = node is up + BFS powered
      Orange = node is up but disconnected
      Yellow = substation / main grid
    """
    powered = get_powered_nodes(network, down_nodes)

    G = nx.Graph()
    pos = {}
    node_colors = []

    for bus in network.buses.index:
        if bus in ["MainPowerGrid", "LocalSubstation"]:
            # Place them near each other for clarity
            pos[bus] = (network.buses.x.get(bus, -50.0), network.buses.y.get(bus, -50.0))
            node_colors.append("yellow")
            G.add_node(bus)
            continue

        G.add_node(bus)
        xval = network.buses.x.get(bus, 0.0)
        yval = network.buses.y.get(bus, 0.0)
        pos[bus] = (xval, yval)

        if bus in down_nodes:
            node_colors.append("red")
        else:
            if bus in powered:
                node_colors.append("green")
            else:
                node_colors.append("orange")

    # Add lines
    for line_name in network.lines.index:
        b0 = network.lines.at[line_name, "bus0"]
        b1 = network.lines.at[line_name, "bus1"]
        G.add_edge(b0, b1)

    # Add transformers
    for trafo_name in network.transformers.index:
        b0 = network.transformers.at[trafo_name, "bus0"]
        b1 = network.transformers.at[trafo_name, "bus1"]
        G.add_edge(b0, b1)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", font_size=8)
    plt.title(f"Power Network at T={time_step}")
    plt.show()

############################
# CSV LOGGING
############################
def initialize_csv_log(network, csv_file="node_stats.csv"):
    """
    Prepare an in-memory dict for storing v_mag/time data, plus create the CSV header.
    Each row = node, columns = time steps T=1, T=2, ...
    """
    voltages = {bus: [] for bus in network.buses.index}
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Node"]
        writer.writerow(header)
        # Each subsequent row: [BusName]
        for bus in network.buses.index:
            writer.writerow([bus])
    return voltages

def append_csv_column(network, voltages, t, csv_file="node_stats.csv"):
    """
    After PF, store bus voltages in voltages dict, append to CSV as a new column T=t.
    """
    # If snapshots empty, define them
    if len(network.snapshots) == 0:
        network.set_snapshots(["now"])

    # Ensure we have PF results
    run_power_flow(network)

    # Check if there's bus voltage data
    if network.buses_t.v_mag_pu.empty:
        print("No bus voltage data. Skipping CSV column.")
        return

    last_v = network.buses_t.v_mag_pu.iloc[-1]
    for bus in network.buses.index:
        voltages[bus].append(last_v.get(bus, 0.0))

    # read current CSV -> append a new column
    with open(csv_file, "r", newline="") as f:
        rows = list(csv.reader(f))

    # first row is ["Node", T=1, T=2, ...]
    rows[0].append(f"T={t}")
    # subsequent rows each correspond to a bus
    bus_index_map = {}
    for i in range(1, len(rows)):
        bus_name = rows[i][0]
        bus_index_map[bus_name] = i

    for bus in network.buses.index:
        vval = last_v.get(bus, 0.0)
        row_idx = bus_index_map[bus]
        rows[row_idx].append(f"{vval:.3f}")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

############################
# LARGE-SCALE / PARTITION FAILURES
############################
def simulate_local_partition_failure(network, down_nodes, center_bus=None, depth=2):
    """
    Find a BFS subgraph of given 'depth' from 'center_bus' and mark them as down.
    If center_bus not given, pick a random bus that isn't down or the substation.
    """
    # Potential centers exclude substation + main grid + already down
    candidates = [
        b for b in network.buses.index
        if b not in ["MainPowerGrid", "LocalSubstation"] and b not in down_nodes
    ]
    if not candidates:
        print("No candidates for local partition failure.")
        return

    if center_bus is None:
        center_bus = random.choice(candidates)

    # Build a graph ignoring currently down nodes
    G = nx.Graph()
    for bus in network.buses.index:
        if bus not in down_nodes:
            G.add_node(bus)
    # Add edges
    for line_name in network.lines.index:
        b0 = network.lines.at[line_name, "bus0"]
        b1 = network.lines.at[line_name, "bus1"]
        if b0 not in down_nodes and b1 not in down_nodes:
            G.add_edge(b0, b1)
    # Also add transformers
    for trafo_name in network.transformers.index:
        b0 = network.transformers.at[trafo_name, "bus0"]
        b1 = network.transformers.at[trafo_name, "bus1"]
        if b0 not in down_nodes and b1 not in down_nodes:
            G.add_edge(b0, b1)

    if center_bus not in G:
        print(f"Center bus {center_bus} is not in the BFS graph. Possibly down or missing.")
        return

    # BFS for 'depth' levels
    # We'll track the level of each node from center_bus
    levels = {n: None for n in G.nodes}
    levels[center_bus] = 0
    queue = [center_bus]

    while queue:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if levels[neighbor] is None:
                levels[neighbor] = levels[current] + 1
                if levels[neighbor] < depth:
                    queue.append(neighbor)

    # Mark all BFS nodes within 'depth' as down
    fail_group = []
    for b in G.nodes:
        if levels[b] is not None and levels[b] <= depth:
            fail_group.append(b)

    print(f"\n💥 Local partition fail from center {center_bus}, depth={depth}, failing: {fail_group}")
    for fn in fail_group:
        down_nodes.add(fn)
        set_node_down(network, fn)

############################
# MAIN SIMULATION
############################
def run_simulation():
    network_file = "osm.net.xml"
    traffic_light_nodes = get_traffic_lights_from_sumo(network_file)
    road_edges = get_road_edges_from_sumo(network_file)

    network, sumo_to_label, label_to_sumo = create_power_network(traffic_light_nodes, road_edges, feeders=3)

    total_steps = 10
    down_nodes = set()
    voltages = initialize_csv_log(network, "node_stats.csv")

    for t in range(1, total_steps + 1):
        print(f"\n=== Time Step {t} ===")

        # random single node failure
        if random.random() < 0.1:
            candidates = [
                b for b in network.buses.index
                if b not in ["MainPowerGrid", "LocalSubstation"] and b not in down_nodes
            ]
            if candidates:
                fail_node = random.choice(candidates)
                print(f"Failing node {fail_node}!")
                down_nodes.add(fail_node)
                set_node_down(network, fail_node)

        # BFS-based local partition failure
        if random.random() < 0.4:
            simulate_local_partition_failure(network, down_nodes, depth=2)

        # random node recovery
        if random.random() < 0.1 and down_nodes:
            recov = random.choice(list(down_nodes))
            print(f"Recovering node {recov}!")
            down_nodes.remove(recov)
            set_node_up(network, recov)

        run_power_flow(network)
        append_csv_column(network, voltages, t, "node_stats.csv")
        visualize_network_state(network, down_nodes, time_step=t)
        time.sleep(1)

    print("\nSimulation ended. Check node_stats.csv for logs.")


if __name__ == "__main__":
    run_simulation()