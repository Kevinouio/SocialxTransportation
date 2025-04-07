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
import matplotlib.image as mpimg
import networkx as nx
import pandas as pd
import numpy as np
import math
import logging
from powerNetworkGen import (
    get_traffic_lights_from_sumo,
    get_road_edges_from_sumo,
    create_power_network,
    build_networkx_graph,
    get_powered_nodes,
    set_node_down,
    set_node_up,
    check_network_connectivity,
    check_impedance,
    add_buildings_from_poly
)
logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("pypsa.pf").setLevel(logging.CRITICAL)
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


############################
# VISUALIZATION
############################

def connect_buildings_to_grid(network):
    """
    Connects all building buses (0.4kV) to the closest existing nodes
    in the main power network (20kV) using transformers.
    """
    # Get all building buses and main network buses
    building_buses = network.buses[network.buses.v_nom == 0.4].index
    main_buses = network.buses[network.buses.v_nom == 20].index

    # Dictionary to track existing connections to avoid duplicates
    existing_connections = set()

    for building in building_buses:
        # Skip already connected buildings
        if f"Trafo_{building}" in network.transformers.index:
            continue

        min_dist = float('inf')
        nearest_bus = None

        # Find closest main network bus
        for main_bus in main_buses:
            try:
                dx = network.buses.at[main_bus, 'x'] - network.buses.at[building, 'x']
                dy = network.buses.at[main_bus, 'y'] - network.buses.at[building, 'y']
                dist = math.hypot(dx, dy)

                if dist < min_dist and (main_bus, building) not in existing_connections:
                    min_dist = dist
                    nearest_bus = main_bus
            except KeyError:
                continue

        if nearest_bus:
            # Add transformer between main network and building
            network.add(
                "Transformer",
                name=f"Trafo_{building}",
                bus0=nearest_bus,
                bus1=building,
                x=0.05,  # 5% reactance
                s_nom=0.1,  # 100 kVA
                tap_ratio=1.0
            )
            existing_connections.add((nearest_bus, building))
            print(f"Connected {building} to {nearest_bus} (distance: {min_dist:.1f}m)")


def visualize_network_state_poly(network, down_nodes, time_step=0):
    plt.figure(figsize=(14, 10))
    G = nx.Graph()

    # Add nodes with relative positions
    all_x = []
    all_y = []
    for bus in network.buses.index:
        if bus in down_nodes:
            continue
        x = network.buses.at[bus, 'x']
        y = network.buses.at[bus, 'y']
        all_x.append(x)
        all_y.append(y)
        node_size = 300 if 'BLD' in bus else 100
        node_color = 'blue' if 'BLD' in bus else 'green'
        G.add_node(bus, size=node_size, color=node_color)

    # Calculate plot center
    plot_center_x = (max(all_x) + min(all_x)) / 2
    plot_center_y = (max(all_y) + min(all_y)) / 2

    # Create relative positions (meters from center)
    pos = {
        bus: (
            network.buses.at[bus, 'x'] - plot_center_x,
            network.buses.at[bus, 'y'] - plot_center_y
        )
        for bus in G.nodes
    }

    # Draw nodes and edges
    nx.draw(G, pos, node_size=[G.nodes[bus]['size'] for bus in G.nodes],
            node_color=[G.nodes[bus]['color'] for bus in G.nodes],
            edge_color='gray', with_labels=True, font_size=8)

    # Use raw UTM coordinates
    pos = {bus: (network.buses.at[bus, 'x'], network.buses.at[bus, 'y'])
           for bus in G.nodes()}

    # Set axis labels in kilometers
    # Fixed version
    node_list = list(G.nodes())
    all_x = network.buses.x[node_list]
    all_y = network.buses.y[node_list]

    plt.xlim(min(all_x) - 100, max(all_x) + 100)  # 100m buffer
    plt.ylim(min(all_y) - 100, max(all_y) + 100)

    plt.gca().set_aspect('equal')
    plt.xlabel("UTM Easting (m)")
    plt.ylabel("UTM Northing (m)")

    plt.title(f"Network State - Hour {time_step + 1}")
    plt.savefig(f"network_visuals/hour_{time_step + 1:02d}.png")
    plt.close()


# Updated visualize_network_state (add transformers)
def visualize_network_state(network, down_nodes, time_step=0):
    os.makedirs("network_visuals", exist_ok=True)
    plt.figure(figsize=(14, 10))
    G = nx.Graph()

    # Add nodes
    for bus in network.buses.index:
        node_size = 300 if 'BLD' in bus else 100
        node_color = 'blue' if 'BLD' in bus else ('red' if bus in down_nodes else 'green')
        G.add_node(bus, size=node_size, color=node_color)

    # Add both lines AND transformers
    # Lines
    for line in network.lines.index:
        b0, b1 = network.lines.at[line, 'bus0'], network.lines.at[line, 'bus1']
        if pd.notna(b0) and pd.notna(b1):
            G.add_edge(b0, b1)

    # Transformers (critical addition)
    for trafo in network.transformers.index:
        b0, b1 = network.transformers.at[trafo, 'bus0'], network.transformers.at[trafo, 'bus1']
        if pd.notna(b0) and pd.notna(b1):
            G.add_edge(b0, b1)

    # Get positions
    pos = {bus: (network.buses.at[bus, 'x'], network.buses.at[bus, 'y'])
           for bus in G.nodes()}

    # Draw with styles
    nx.draw(
        G, pos,
        node_size=[G.nodes[bus]['size'] for bus in G.nodes()],
        node_color=[G.nodes[bus]['color'] for bus in G.nodes()],
        edge_color='gray',
        with_labels=True,
        font_size=8
    )

    # Set axis limits
    all_x = [network.buses.at[bus, 'x'] for bus in G.nodes()]
    all_y = [network.buses.at[bus, 'y'] for bus in G.nodes()]
    plt.xlim(min(all_x) - 100, max(all_x) + 100)
    plt.ylim(min(all_y) - 100, max(all_y) + 100)

    plt.title(f"Combined Network State - Hour {time_step + 1}\n"
              f"Traffic Nodes: {len([b for b in network.buses.index if 'N' in b])} | "
              f"Buildings: {len([b for b in network.buses.index if 'BLD' in b])}")
    plt.savefig(f"network_visuals/combined_hour_{time_step + 1:02d}.png")
    plt.close()

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
# DEBUG UTILITIES
############################
def debug_node_coordinates(network):
    """Prints coordinates and details of all network nodes"""
    print("\n=== NODE COORDINATE DEBUG ===")
    print(f"{'Node Name':<15} | {'X (m)':<12} | {'Y (m)':<12} | {'Voltage (kV)':<12} | {'Type':<10}")
    print("-" * 70)

    building_count = 0
    traffic_node_count = 0
    other_count = 0

    for bus in network.buses.index:
        x = network.buses.at[bus, 'x']
        y = network.buses.at[bus, 'y']
        v_nom = network.buses.at[bus, 'v_nom']

        if 'BLD' in bus:
            node_type = 'Building'
            building_count += 1
        elif bus.startswith('N') and bus[1:].isdigit():
            node_type = 'Traffic Node'
            traffic_node_count += 1
        else:
            node_type = 'Infrastructure'
            other_count += 1

        print(f"{bus:<15} | {x:<12.2f} | {y:<12.2f} | {v_nom:<12.1f} | {node_type:<10}")

    print("\n=== SUMMARY ===")
    print(f"Total Nodes: {len(network.buses.index)}")
    print(f"- Traffic Nodes: {traffic_node_count}")
    print(f"- Buildings: {building_count}")
    print(f"- Other Infrastructure: {other_count}")
    print("=" * 70 + "\n")
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


def create_power_network_from_poly(poly_file="osm.poly.xml"):
    """
    Creates a PyPSA network from SUMO building polygons with:
    - Main grid (20 kV) -> Substation (20 kV) -> Buildings (0.4 kV via transformers)
    - Each building has a small load
    - Radial connection through transformers
    """
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    import pyproj
    import numpy as np

    network = pypsa.Network()

    # ======================
    # 1. Core Infrastructure
    # ======================
    # Main power grid (slack bus)
    network.add("Bus", "MainPowerGrid", v_nom=20, x=50, y=50)
    network.add("Generator", "MainGenerator", bus="MainPowerGrid",
                p_nom=100, control="Slack")

    # Local substation
    network.add("Bus", "LocalSubstation", v_nom=20, x=60, y=50)
    network.add("Line", "Line_Grid_Sub", bus0="MainPowerGrid",
                bus1="LocalSubstation", x=0.1, r=0.01, s_nom=100)

    # ======================
    # 2. Coordinate System Setup
    # ======================
    tree = ET.parse(poly_file)
    root = tree.getroot()
    location = root.find('location')

    # Validate projection parameters
    if location is None:
        raise ValueError("Missing <location> element in poly file")

    proj_params = location.get('projParameter', '')
    if '+proj=utm' not in proj_params or '+zone=' not in proj_params:
        raise ValueError("Only UTM projections are supported")

    # Extract UTM zone and hemisphere
    zone = proj_params.split('+zone=')[1].split()[0]
    hemisphere = 'north' if '+north' in proj_params else 'south'
    epsg_code = 32600 + int(zone) if hemisphere == 'north' else 32700 + int(zone)

    # Create coordinate transformer
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{epsg_code}",
        'EPSG:3857'  # Web Mercator
    )

    # ======================
    # 3. Process Buildings
    # ======================
    building_count = 0
    skipped_buildings = 0

    for poly_elem in root.findall('poly'):
        if not poly_elem.get('type', '').startswith('building'):
            continue

        # Parse UTM coordinates directly
        shape_str = poly_elem.get('shape')
        coords = [tuple(map(float, p.split(','))) for p in shape_str.split()]

        try:
            # Create polygon from raw UTM coordinates
            utm_poly = Polygon(coords)
            if not utm_poly.is_valid:
                continue
            centroid = utm_poly.centroid
        except:
            continue

        # Add bus with UTM coordinates
        building_id = f"BLD{poly_elem.get('id')}"
        network.add("Bus", building_id, v_nom=0.4,
                    x=centroid.x, y=centroid.y)  # Direct UTM values
        # Add residential load (2-5 kW)
        network.add("Load", f"Load_{building_id}", bus=building_id,
                    p_set=np.random.uniform(0.002, 0.005))  # MW

        # Connect to substation via transformer (20 kV -> 0.4 kV)
        network.add("Transformer", f"Trafo_{building_id}",
                    bus0="LocalSubstation", bus1=building_id,
                    x=0.05, s_nom=0.1)

        building_count += 1

    # ======================
    # 5. Final Validation
    # ======================
    if building_count == 0:
        raise RuntimeError(f"No valid buildings found in {poly_file}")

    print(f"Network created with:"
          f"\n- {building_count} buildings"
          f"\n- {skipped_buildings} invalid buildings skipped"
          f"\n- Base load: {len(network.loads)} loads")

    return network


############################
# MANUAL NODE FAILURE
############################
def trigger_manual_failure(network, down_nodes, failure_duration=1000):
    """
    Allows user to manually select a node to fail from available nodes.
    Returns updated down_nodes dictionary.
    """
    # Get eligible nodes
    candidates = [b for b in network.buses.index
                  if b not in ["MainPowerGrid", "LocalSubstation"]
                  and b not in down_nodes]

    if not candidates:
        print("No nodes available for manual failure")
        return down_nodes

    print("\n=== Available Nodes ===")
    print("Main Infrastructure Nodes:")
    print([b for b in candidates if not b.startswith(('N', 'BLD'))])

    print("\nTraffic Light Nodes (N*):")
    print([b for b in candidates if b.startswith('N')])

    print("\nBuilding Nodes (BLD*):")
    print([b for b in candidates if b.startswith('BLD')])

    while True:
        node_id = input("\nEnter node ID to fail (or 'cancel'): ").strip()
        if node_id.lower() == 'cancel':
            return down_nodes
        if node_id in candidates:
            down_nodes[node_id] = failure_duration
            set_node_down(network, node_id)
            print(f"Node {node_id} scheduled for failure")
            return down_nodes
        print(f"Invalid node ID. Must be one of: {candidates}")


def simulate_local_failure(network, original_graph, failed_node, depth, failure_chance=0.3):
    """Fail nearby nodes within network distance (hops)"""
    # Track nodes within specified depth
    affected_nodes = set()
    visited = {failed_node: 0}
    queue = [(failed_node, 0)]  # (node, current_depth)

    while queue:
        node, current_depth = queue.pop(0)
        if current_depth >= depth:
            continue

        for neighbor in original_graph.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = current_depth + 1
                queue.append((neighbor, current_depth + 1))
                affected_nodes.add(neighbor)

    # Filter out protected nodes
    candidates = [n for n in affected_nodes
                  if n not in ["MainPowerGrid", "LocalSubstation"]]

    # Fail random subset
    to_fail = []
    for node in candidates:
        if random.random() < failure_chance:
            to_fail.append(node)

    # Apply failures
    for node in to_fail:
        set_node_down(network, node)

    return to_fail

############################
# MAIN SIMULATION
############################
def run_combined_simulation():
    """Simulation combining traffic light network and poly-file buildings"""
    # 1. Create base traffic light network
    network_file = "osm.net.xml"
    traffic_light_nodes = get_traffic_lights_from_sumo(network_file)
    road_edges = get_road_edges_from_sumo(network_file)
    network, sumo_to_label, label_to_sumo, _ = create_power_network(
        traffic_light_nodes,
        road_edges,
        feeders=3
    )

    # 2. Add buildings from poly file
    add_buildings_from_poly(network, "osm.poly.xml")

    # 3. Connect buildings to closest traffic nodes
    connect_buildings_to_grid(network)
    debug_node_coordinates(network)

    # 4. Infrastructure tuning
    network.generators.at["MainGenerator", "p_nom"] = 150  # MW
    network.lines.at["Line_MPG_LS", "s_nom"] = 100  # MW capacity

    # 5. Set up temporal parameters
    total_steps = 24
    network.set_snapshots(range(total_steps))

    # Create realistic EV load profile (peaks at midday)
    time_array = np.linspace(0, 2 * np.pi, total_steps)
    ev_load_profile_mw = 30 + 100 * np.cos(time_array - np.pi / 2) ** 2  # 10-50MW load
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

    # 7. Initialize failure tracking
    failed_buses = set()
    voltage_threshold = 0.88  # ANSI standard

    # 8. Main simulation loop
    for t in network.snapshots:
        print(f"\n=== Combined Network - Hour {t + 1} ===")

        try:
            run_power_flow(network)
        except Exception as e:
            print(f"Power flow failed: {str(e)[:100]}")
            failed_buses.update(network.buses.index)
        else:
            # Voltage violations
            low_voltage = network.buses_t.v_mag_pu.loc[t] < voltage_threshold
            failed_buses.update(low_voltage[low_voltage].index.tolist())

        # Visualization
        visualize_network_state(network, failed_buses, t)

    # 9. Save results
    network.buses_t.v_mag_pu.T.to_csv("combined_voltage_log.csv")
    print("\nCombined simulation complete")


############################
# MANUAL NODE FAILURE SIMULATION
############################
def run_manual_failure_simulation():
    """Run power network simulation with manual node failure selection and voltage logging"""
    # Create network
    network_file = "osm.net.xml"
    traffic_light_nodes = get_traffic_lights_from_sumo(network_file)
    road_edges = get_road_edges_from_sumo(network_file)
    network, sumo_to_label, label_to_sumo, original_graph = create_power_network(traffic_light_nodes, road_edges, feeders=3)
    add_buildings_from_poly(network, "osm.poly.xml")

    # Initialize logging and tracking
    voltage_log = initialize_power_flow_log_df(network)
    current_step = 0
    failed_nodes = {}  # {node_id: remaining_downtime_steps}
    failure_duration = 5  # Default recovery after 5 steps

    try:
        while True:
            # 1. Automatic node recovery check
            to_remove = []
            for node in list(failed_nodes.keys()):
                failed_nodes[node] -= 1
                if failed_nodes[node] <= 0:
                    to_remove.append(node)

            for node in to_remove:
                del failed_nodes[node]
                set_node_up(network, node, failed_nodes)
                print(f"\n🔌 Node {node} automatically restored after downtime")

            # 2. Run power flow and log voltages
            try:
                run_power_flow(network)
                voltage_log = append_power_flow_log_df(network, voltage_log, current_step)
            except Exception as e:
                print(f"Power flow calculation failed: {str(e)[:50]}")

            # 3. Visualization
            visualize_network_state(network, failed_nodes.keys(), current_step)

            # 4. User interface
            print("\n" + "=" * 40)
            print(f"STEP {current_step} - Active failures: {failed_nodes}")
            print("[f]ail node, [r]ecover node, [v]iew log, [q]uit")

            action = input("Action: ").lower()

            # 5. Handle user actions
            if action == 'f':
                available = [n for n in network.buses.index
                             if n not in failed_nodes
                             and n not in ["MainPowerGrid", "LocalSubstation"]]

                if not available:
                    print("No nodes available for failure")
                    continue

                print("\nAvailable nodes:", available)
                node = input("Node ID to fail: ").strip()

                if node in available:
                    try:
                        duration = int(
                            input(f"Failure duration (current default {failure_duration}): ") or failure_duration)
                    except ValueError:
                        duration = failure_duration

                    # Set node failure
                    failed_nodes[node] = duration
                    set_node_down(network, node)

                    # Optional cascading failure
                    try:
                        depth = int(input("Cascade failure depth (0-3): ") or 0)
                        depth = max(0, min(3, depth))
                        if depth > 0:
                            local_failures = simulate_local_failure(network, original_graph, node, depth)
                            for fn in local_failures:
                                failed_nodes[fn] = duration  # Same duration as main failure
                    except:
                        pass

            elif action == 'r' and failed_nodes:
                print("Failed nodes:", list(failed_nodes.keys()))
                node = input("Node ID to recover: ").strip()
                if node in failed_nodes:
                    del failed_nodes[node]
                    set_node_up(network, node, failed_nodes)
                    print(f"Node {node} manually restored")

            elif action == 'v':
                print("\nVoltage log summary:")
                print(voltage_log.tail(3))

            elif action == 'q':
                break

            current_step += 1

    finally:
        # Save final state
        save_power_flow_log_df(voltage_log, "manual_simulation_voltages.csv")
        print("\nFinal network state:")
        debug_node_coordinates(network)

# Update the main block to use this new simulation
if __name__ == "__main__":
    run_combined_simulation()


