import os
import random
import time
import copy
import csv
import pypsa
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

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
# EV CHARGING STATION
############################

def add_ev_charging_station(
    network: pypsa.Network,
    station_name: str,
    connect_to_bus: str,
    line_length_km: float = 1.0,
    line_reactance_per_km: float = 0.01,
    line_resistance_per_km: float = 0.001,
    line_rating_mva: float = 5.0,
    ev_load_profile_kw: pd.Series = None,
    bus_coords: tuple = (0.0, 0.0),
):
    """
    Dynamically adds an EV charging station to the given PyPSA network by:
      1. Creating a new bus for the EV station.
      2. Creating a new line from the existing bus (connect_to_bus) to this new EV bus.
      3. Adding a load with a time-varying (or static) demand profile at the new EV bus.

    Parameters
    ----------
    network : pypsa.Network
        The existing network object to which the station will be added.
    station_name : str
        Base name for the bus, line, and load components (e.g., "EV_Station").
    connect_to_bus : str
        The name of the bus in the existing network to which the EV station will connect.
    line_length_km : float
        The length of the connecting line in kilometers.
    line_reactance_per_km : float
        Per-km reactance for the line (in p.u. or the appropriate PyPSA internal unit).
    line_resistance_per_km : float
        Per-km resistance for the line (in p.u. or the appropriate PyPSA internal unit).
    line_rating_mva : float
        The MVA rating (thermal limit) of the line connecting the station.
    ev_load_profile_kw : pd.Series, optional
        A time series (indexed by `network.snapshots`) representing the EV load in kW.
        If None, defaults to a constant 100 kW.
    bus_coords : tuple
        (x, y) coordinates of the new EV bus for visualization or geo-referencing.

    Returns
    -------
    None
        Modifies the given `network` in place.
    """
    if ev_load_profile_kw is None:
        # Default to a constant 100 kW if no profile is provided
        ev_load_profile_kw = pd.Series(100.0, index=network.snapshots)

    # Convert from kW to MW for PyPSA (common practice)
    ev_load_profile_mw = ev_load_profile_kw / 1000.0

    # 1. Create a new bus for the EV station
    ev_bus_name = f"{station_name}_Bus"
    network.add(
        "Bus",
        ev_bus_name,
        x=bus_coords[0],
        y=bus_coords[1],
    )

    # 2. Create a new line connecting the existing bus to the new EV bus
    ev_line_name = f"{station_name}_Line"
    network.add(
        "Line",
        ev_line_name,
        bus0=connect_to_bus,
        bus1=ev_bus_name,
        length=line_length_km,
        x=line_reactance_per_km * line_length_km,
        r=line_resistance_per_km * line_length_km,
        s_nom=line_rating_mva,  # Nominal rating in MVA
    )

    # 3. Add a new load (with a time-varying p_set) for the EV station
    ev_load_name = f"{station_name}_Load"
    network.add(
        "Load",
        ev_load_name,
        bus=ev_bus_name,
        p_set=ev_load_profile_mw,  # Time series in MW
        q_set=0.0,                 # Set Q to 0 or your desired reactive load
    )

def add_ev_charging_station(network, station_name, connect_to_bus,
                            line_length_km=1.0,
                            line_x_per_km=0.01,
                            line_r_per_km=0.001,
                            line_rating_mva=5.0):
    """
    Adds a simple EV charging station to the network by:
      - Creating a new bus
      - Creating a line connecting to 'connect_to_bus'
      - Adding a Load representing the EV station's demand
    """
    ev_bus_name = f"{station_name}_Bus"
    ev_line_name = f"{station_name}_Line"
    ev_load_name = f"{station_name}_Load"

    # 1. Add the bus
    network.add(
        "Bus",
        ev_bus_name,
        x=0.0,  # optionally set geographic x,y if you want to visualize
        y=0.0
    )

    # 2. Add a line to connect the EV bus to the existing bus
    network.add(
        "Line",
        ev_line_name,
        bus0=connect_to_bus,
        bus1=ev_bus_name,
        length=line_length_km,
        x=line_x_per_km * line_length_km,
        r=line_r_per_km * line_length_km,
        s_nom=line_rating_mva
    )

    # 3. Add the EV load (initially set it to something small or zero)
    network.add(
        "Load",
        ev_load_name,
        bus=ev_bus_name,
        p_set=0.0,   # in MW
        q_set=0.0
    )

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
# LOGGING
############################
def initialize_power_flow_log():
    """
    Create an empty DataFrame with columns we'll need:
      - time (int or float)
      - bus (string)
      - voltage (float)
    """
    df = pd.DataFrame(columns=["time", "bus", "voltage_pu"])
    return df


def log_power_flow_snapshot(network, df, time_step):
    """
    Run power flow, then append each bus's voltage to df.
    Returns the updated DataFrame.
    """
    # Ensure we do a power flow
    run_power_flow(network)

    # If the network has at least one snapshot, we can read the last row of 'v_mag_pu'
    if not network.buses_t.v_mag_pu.empty:
        v_series = network.buses_t.v_mag_pu.iloc[-1]  # last row
        # v_series is typically indexed by bus name, e.g. v_series["N1"] = 0.99, etc.

        # Build a small DataFrame for this step
        # time, bus, voltage_pu
        step_data = []
        for bus in network.buses.index:
            bus_voltage = v_series.get(bus, 0.0)
            step_data.append([time_step, bus, bus_voltage])

        step_df = pd.DataFrame(step_data, columns=["time", "bus", "voltage_pu"])
        df = pd.concat([df, step_df], ignore_index=True)

    else:
        print("No voltage data available for this snapshot.")

    return df

def save_power_flow_log(df, csv_file="node_stats.csv"):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(csv_file, index=False)
    print(f"Saved power flow log to {csv_file}")


############################
# MAIN SIMULATION
############################
def run_simulation():
    network_file = "../osm.net.xml"

    # Assuming you have these helper functions already:
    traffic_light_nodes = get_traffic_lights_from_sumo(network_file)
    road_edges = get_road_edges_from_sumo(network_file)

    # Create the network
    network, sumo_to_label, label_to_sumo = create_power_network(
        traffic_light_nodes,
        road_edges,
        feeders=3
    )

    # --------------------------------------------------
    # A) Add the EV charging station to the network
    #    Suppose we connect it to an existing bus called "Bus0"
    #    (or any bus name that exists in the created network).
    # --------------------------------------------------
    add_ev_charging_station(
        network=network,
        station_name="EVStation1",
        connect_to_bus="Bus0",  # Adjust to a real bus name in your network
        line_length_km=0.5,
        line_x_per_km=0.01,
        line_r_per_km=0.001,
        line_rating_mva=5.0
    )

    total_steps = 10
    down_nodes = set()

    # Initialize an empty DataFrame to store logs
    df_power_log = initialize_power_flow_log()

    for t in range(1, total_steps + 1):
        print(f"\n=== Time Step {t} ===")

        # -----------------------------------------
        # B) Update the EV load for the current time step
        # -----------------------------------------
        # For demonstration, let's do something simple: ramp load linearly
        # from 0 MW at step 1 to 0.3 MW at step 10 (i.e., 300 kW).
        p_mw = 0.3 * (t / total_steps)

        # Update the load in the PyPSA network
        # The load name is <station_name>_Load => "EVStation1_Load"
        network.loads.at["EVStation1_Load", "p_set"] = p_mw

        # ... do your random node failures, BFS partitions, recoveries, etc. ...
        # e.g., randomly pick some buses or lines to disable, then re-enable them.

        # 2) Run PF and log the voltages
        df_power_log = log_power_flow_snapshot(network, df_power_log, t)

        # 3) (Optional) visualize the network state
        visualize_network_state(network, down_nodes, time_step=t)

        # Possibly sleep or do other timing steps, e.g. time.sleep(1)

    # 4) Save the CSV once the entire simulation finishes
    save_power_flow_log(df_power_log, "../node_stats.csv")

    print("\nSimulation ended. Check node_stats.csv for logs.")


if __name__ == "__main__":
    run_simulation()