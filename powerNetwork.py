import os
import random
import time
import copy
import csv
import pypsa
import matplotlib

matplotlib.use("TkAgg")  # or "Agg" or "TkAgg"
import networkx
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
    set_node_up,
    check_network_connectivity,
    check_impedance
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
        q_set=0.0,  # Set Q to 0 or your desired reactive load
    )


#
# VISUALIZATION
############################
def visualize_network_state(network, down_nodes, time_step=0):
    """Modified visualization to handle disconnected lines and save to directory"""
    import os
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs("network_visuals", exist_ok=True)

    G = nx.Graph()

    # Add nodes
    for bus_name in network.buses.index:
        G.add_node(bus_name)

    # Add edges only for active lines
    for line_name in network.lines.index:
        bus0 = network.lines.at[line_name, "bus0"]
        bus1 = network.lines.at[line_name, "bus1"]

        # Skip lines with disconnected buses
        if pd.isna(bus0) or pd.isna(bus1):
            continue

        if bus0 not in down_nodes and bus1 not in down_nodes:
            G.add_edge(bus0, bus1)

    # Get positions from network coordinates
    pos = {bus: (network.buses.at[bus, "x"], network.buses.at[bus, "y"])
           for bus in G.nodes()}

    # Create node colors
    node_colors = ["red" if node in down_nodes else "green" for node in G.nodes()]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=200,
            node_color=node_colors, edge_color="gray",
            font_size=8, font_weight="bold")

    plt.title(f"Network State - Hour {time_step + 1}")
    plt.savefig(f"network_visuals/network_t{time_step:02d}.png")
    plt.close()


############################
# CSV LOGGING
############################



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
def initialize_power_flow_log_df(network):
    """
    Create an empty DataFrame with bus names as the index.
    Each column will represent a time step (e.g., "T=1", "T=2", ...).
    """
    df = pd.DataFrame(index=network.buses.index)
    return df


def append_power_flow_log_df(network, df, t):
    """
    Runs a power flow on the network, extracts the voltage magnitudes,
    and appends them as a new column labeled "T=t" in the DataFrame.
    """
    run_power_flow(network)
    if network.buses_t.v_mag_pu.empty:
        print(f"No voltage data available at time T={t}.")
        return df
    last_v = network.buses_t.v_mag_pu.iloc[-1]
    df[f"T={t}"] = last_v
    return df


def save_power_flow_log_df(df, csv_file="node_stats.csv"):
    """
    Saves the DataFrame to a CSV file.
    """
    df.to_csv(csv_file, index=True)
    print(f"Saved power flow log to {csv_file}")


# Add this helper function to preserve original connections
def cache_original_connections(network):
    return {
        "lines": network.lines[["bus0", "bus1"]].copy(),
        "buses": network.buses[["x", "y"]].copy()
    }

############################
# MAIN SIMULATION
############################
import numpy as np


def run_simulation():
    network_file = "osm.net.xml"

    traffic_light_nodes = get_traffic_lights_from_sumo(network_file)
    road_edges = get_road_edges_from_sumo(network_file)

    # Create main network with realistic parameters
    network, sumo_to_label, label_to_sumo = create_power_network(
        traffic_light_nodes,
        road_edges,
        feeders=3
    )

    # Make infrastructure marginally adequate
    network.generators.at["MainGenerator", "p_nom"] = 80  # MW
    network.lines.at["Line_MPG_LS", "s_nom"] = 60  # MW capacity

    # Set up snapshots for 24-hour simulation
    total_steps = 24
    network.set_snapshots(range(total_steps))

    # Create realistic EV load profile (peaks at midday)
    time_array = np.linspace(0, 2 * np.pi, total_steps)
    ev_load_profile_mw = 10 + 70 * np.cos(time_array - np.pi / 2) ** 2  # 10-50MW load
    ev_load_profile = pd.Series(ev_load_profile_mw * 1000,  # kW
                                index=network.snapshots)

    # Add EV station to existing infrastructure
    n30_coords = (network.buses.at["N30", "x"], network.buses.at["N30", "y"])
    add_ev_charging_station(
        network=network,
        station_name="EVStation1",
        connect_to_bus="N30",
        ev_load_profile_kw=ev_load_profile,
        bus_coords=(n30_coords[0] + 2, n30_coords[1] + 2),
        line_rating_mva=35,  # Keep existing line rating
        line_resistance_per_km=0.003,  # Realistic values for 20kV lines
        line_reactance_per_km=0.007
    )
    original_network_state = cache_original_connections(network)
    # Failure tracking with recovery potential
    # Modified failure handling in run_simulation()
    failed_lines = set()
    failed_buses = set()
    voltage_threshold = 0.88  # ANSI minimum voltage standard

    for t in network.snapshots:
        print(f"\n=== Hour {t + 1} ===")
        print(f"EV Load: {ev_load_profile_mw[t]:.1f} MW")

        # Reset previous line failures (optional: add recovery logic here)
        # network.lines[["bus0", "bus1"]] = original_line_connections

        try:
            run_power_flow(network)
        except Exception as e:
            print(f"⚡ Power flow failed: {str(e)[:100]}...")
            failed_buses.add("EVStation1_Bus")
            network.buses_t.v_mag_pu.loc[t] = 0  # Mark all buses as failed
        else:
            # Detect voltage violations
            low_voltage = network.buses_t.v_mag_pu.loc[t] < voltage_threshold
            failed_buses.update(low_voltage[low_voltage].index.tolist())

            # Detect line overloads (100% capacity)
            line_loading = (network.lines_t.p0.loc[t].abs() /
                            network.lines.s_nom.replace(0, 1e-6))
            overloaded_lines = line_loading[line_loading > 1.0].index
            failed_lines.update(overloaded_lines)

        # Visualize current state (handles NaN buses automatically)
        visualize_network_state(network, failed_buses, t)

    # Save final results
    voltage_log = network.buses_t.v_mag_pu.copy()
    voltage_log.T.to_csv("voltage_log.csv")


if __name__ == "__main__":
    run_simulation()
