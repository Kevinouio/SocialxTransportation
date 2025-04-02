import xml.etree.ElementTree as ET
import os
import math
import pypsa
import networkx as nx
import numpy as np
os.environ["PROJ_LIB"] = r"C:\Users\kth258\AppData\Local\anaconda3\envs\sot\Library\share\proj"


def get_traffic_lights_from_sumo(network_file):
    """
    Extracts traffic lights from a SUMO .net.xml file as {id: (x, y)}.
    """
    tree = ET.parse(network_file)
    root = tree.getroot()
    traffic_lights = {}

    for junction in root.findall("junction"):
        jtype = junction.get("type")
        if jtype in ["traffic_light", "priority", "unregulated"]:
            node_id = junction.get("id")
            x, y = float(junction.get("x")), float(junction.get("y"))
            traffic_lights[node_id] = (x, y)
    return traffic_lights


def get_road_edges_from_sumo(network_file):
    """
    Extracts edges from SUMO .net.xml as a list of (from_node, to_node).
    """
    tree = ET.parse(network_file)
    root = tree.getroot()
    edges = []
    for edge in root.findall("edge"):
        fr = edge.get("from")
        to = edge.get("to")
        if fr and to:
            edges.append((fr, to))
    return edges


def create_power_network(traffic_light_nodes, road_edges, feeders=2):
    """
    Creates a PyPSA network with:
      - MainPowerGrid (230 kV, slack generator)
      - LocalSubstation (20 kV)
      - Transformer bridging them
      - Multiple feeders connecting substation -> traffic lights
      - Each traffic light bus has a 0.02 MW load

    Updated with more realistic line impedances & reduced generator capacity.
    """
    network = pypsa.Network()

    # Slack bus: main power grid
    network.add("Bus", name="MainPowerGrid", v_nom=230, x=50, y=50)
    # LOWER the nominal capacity to ~50 MW if total load is small
    network.add("Generator", name="MainGenerator", bus="MainPowerGrid", p_nom=200, control="Slack")

    # Local substation at 20 kV
    network.add("Bus", name="LocalSubstation", v_nom=20, x=60, y=50)
    # Transformer 230 -> 20 kV
    network.add(
        "Line",
        "Line_MPG_LS",  # example
        bus0="MainPowerGrid",
        bus1="LocalSubstation",
        length=1.0,
        r=0.5,  # small but non-zero
        x=0.9,
        s_nom=100
    )

    # Assign short labels to traffic lights
    sumo_to_label = {}
    label_to_sumo = {}
    for i, (sumo_id, (x, y)) in enumerate(traffic_light_nodes.items(), start=1):
        label = f"N{i}"
        sumo_to_label[sumo_id] = label
        label_to_sumo[label] = sumo_id

        network.add("Bus", name=label, v_nom=20, x=x, y=y)
        # Each traffic light ~0.02 MW
        network.add("Load", name=f"load_{label}", bus=label, p_set=0.02)

    # Connect local substation to multiple feeders (some random or first 'feeders' nodes)
    all_labels = list(sumo_to_label.values())
    feeder_labels = all_labels[:feeders]
    # Example line data from substation to feeders
    for feed_l in feeder_labels:
        network.add(
            "Line",
            name=f"Feeder_{feed_l}",
            bus0="LocalSubstation",
            bus1=feed_l,
            length=0.1,
            r=0.1,
            x=0.9,
            r_per_length=0.0003,  # More realistic R
            x_per_length=0.0004
        )

    # Connect traffic lights with lines based on road edges
    # Use a bit higher impedances for distribution lines
    r_per_m = 0.0003
    x_per_m = 0.0004
    for (s_from, s_to) in road_edges:
        if s_from in sumo_to_label and s_to in sumo_to_label:
            from_label = sumo_to_label[s_from]
            to_label = sumo_to_label[s_to]
            (x1, y1) = traffic_light_nodes[s_from]
            (x2, y2) = traffic_light_nodes[s_to]
            dist_m = math.hypot(x2 - x1, y2 - y1)
            dist_km = dist_m / 1000.0

            network.add(
                "Line",
                name=f"{from_label}-{to_label}",
                bus0=from_label,
                bus1=to_label,
                length=dist_km,
                x=0.5,
                r=0.9,
                r_per_length=r_per_m,
                x_per_length=x_per_m
            )
    # Right before returning:
    if "MainPowerGrid" in network.buses.index:
        if network.generators.empty or not (network.generators["control"] == "Slack").any():
            network.add(
                "Generator",
                "SlackGen",
                bus="MainPowerGrid",
                p_set=0,
                control="Slack",
                max_p_pu=1e9,
                min_p_pu=-1e9
            )
    else:
        print("WARNING: No 'MainPowerGrid' bus found. Please create or rename a bus to serve as Slack.")

    # Add at the end before return
    original_graph = build_networkx_graph(network, down_nodes=set())

    return network, sumo_to_label, label_to_sumo, original_graph  # Modified return


def build_networkx_graph(network, down_nodes):
    """
    Build a NetworkX graph from PyPSA lines + transformers,
    ignoring any node in 'down_nodes'.
    """
    G = nx.Graph()
    for bus in network.buses.index:
        if bus in down_nodes:
            continue
        G.add_node(bus)

    # Lines
    for line_name in network.lines.index:
        b0 = network.lines.at[line_name, "bus0"]
        b1 = network.lines.at[line_name, "bus1"]
        if b0 not in down_nodes and b1 not in down_nodes:
            G.add_edge(b0, b1)

    # Transformers
    for trafo_name in network.transformers.index:
        b0 = network.transformers.at[trafo_name, "bus0"]
        b1 = network.transformers.at[trafo_name, "bus1"]
        if b0 not in down_nodes and b1 not in down_nodes:
            G.add_edge(b0, b1)

    return G


def get_powered_nodes(network, down_nodes):
    """
    BFS from 'MainPowerGrid' and 'LocalSubstation' to find which nodes are energized.
    If a node is in 'down_nodes', it's forced out of BFS.
    """
    G = build_networkx_graph(network, down_nodes)
    powered = set()
    for source in ["MainPowerGrid", "LocalSubstation"]:
        if source in G:
            for node in nx.bfs_tree(G, source):
                powered.add(node)
    return powered


def set_node_down(network, node):
    # Disconnect lines
    lines_to_disable = network.lines[
        (network.lines.bus0 == node) | (network.lines.bus1 == node)
    ].index
    network.lines.loc[lines_to_disable, "in_service"] = False

    # Disconnect transformers
    trafos_to_disable = network.transformers[
        (network.transformers.bus0 == node) | (network.transformers.bus1 == node)
    ].index
    network.transformers.loc[trafos_to_disable, "in_service"] = False

    # Zero out loads (critical for accurate power flow)
    network.loads.loc[network.loads.bus == node, "p_set"] = 0

    # Optional: Set bus itself as out of service (if supported by PyPSA)
    if "in_service" in network.buses.columns:
        network.buses.loc[node, "in_service"] = False


def set_node_up(network, node):
    """
    Reactivate a node's load (set to 0.02).
    """
    for load_name in network.loads.index:
        if network.loads.at[load_name, "bus"] == node:
            network.loads.at[load_name, "p_set"] = 0.02


def check_network_connectivity(network):
    G = build_networkx_graph(network, down_nodes=[])
    components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"Component {i}: {comp}")


def check_impedance(network):
    print("Line Impedance Values:")
    for line in network.lines.index:
        # If you have r and x defined directly:
        r = network.lines.at[line, "r"] if "r" in network.lines.columns else None
        x = network.lines.at[line, "x"] if "x" in network.lines.columns else None

        length = network.lines.at[line, "length"]
        r_eff = network.lines.r_per_length.get(line, 0) * length
        x_eff = network.lines.x_per_length.get(line, 0) * length
        print(f"Line {line}: effective r={r_eff:.6f}, effective x={x_eff:.6f}")

        # If using per_length values, you might have columns like r_per_length and x_per_length
        r_per_length = network.lines.at[line, "r_per_length"] if "r_per_length" in network.lines.columns else None
        x_per_length = network.lines.at[line, "x_per_length"] if "x_per_length" in network.lines.columns else None

        print(f"Line {line}: r={r} x={x} | r_per_length={r_per_length} x_per_length={x_per_length}")



def add_buildings_from_poly(network, poly_file="osm.poly.xml", max_buildings=50):
    """Add a limited number of randomly selected buildings from poly file"""
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    import random

    tree = ET.parse(poly_file)
    root = tree.getroot()

    # First collect all valid buildings
    all_buildings = []
    for poly_elem in root.findall('poly'):
        if not poly_elem.get('type', '').startswith('building'):
            continue

        shape_str = poly_elem.get('shape')
        if not shape_str or len(shape_str.split()) < 4:
            continue

        try:
            coords = [tuple(map(float, p.split(','))) for p in shape_str.split()]
            poly = Polygon(coords)
            if poly.is_valid:
                all_buildings.append((poly_elem.get('id'), poly.centroid))
        except Exception:
            continue

    # Randomly select up to max_buildings
    selected_buildings = random.sample(all_buildings, min(max_buildings, len(all_buildings)))

    for building_id, centroid in selected_buildings:
        # Add to network
        bus_id = f"BLD{building_id}"
        network.add("Bus",
                   name=bus_id,
                   v_nom=0.4,
                   x=centroid.x,
                   y=centroid.y)

        network.add("Load",
                   name=f"load_{bus_id}",
                   bus=bus_id,
                   p_set=np.random.uniform(0.5, 2.0))  # 500-2000 kW

        # Connect to nearest 20kV node
        min_dist = float('inf')
        nearest_bus = None
        for bus in network.buses.index:
            if network.buses.at[bus, 'v_nom'] == 20:
                dist = math.hypot(network.buses.at[bus, 'x'] - centroid.x,
                                 network.buses.at[bus, 'y'] - centroid.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_bus = bus

        if nearest_bus:
            network.add("Transformer",
                       name=f"trafo_{bus_id}",
                       bus0=nearest_bus,
                       bus1=bus_id,
                       x=0.05,
                       s_nom=5.0)