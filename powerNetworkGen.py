import xml.etree.ElementTree as ET
import os
import math
import pypsa
import networkx as nx

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
    network.add("Bus", name="MainPowerGrid", v_nom=230)
    # LOWER the nominal capacity to ~50 MW if total load is small
    network.add("Generator", name="MainGenerator", bus="MainPowerGrid", p_nom=50, control="Slack")

    # Local substation at 20 kV
    network.add("Bus", name="LocalSubstation", v_nom=20)

    # Transformer 230 -> 20 kV
    network.add(
        "Transformer",
        name="GridTransformer",
        bus0="MainPowerGrid",
        bus1="LocalSubstation",
        s_nom=100,  # MVA rating
        v_nom0=230,
        v_nom1=20,
        x_sc=10,  # ~10%
        r_sc=1  # ~1%
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
                r_per_length=r_per_m,
                x_per_length=x_per_m
            )

    return network, sumo_to_label, label_to_sumo


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
    """
    Mark a node as 'down' by zeroing out its load. Keep the bus in the network for visualization.
    """
    for load_name in network.loads.index:
        if network.loads.at[load_name, "bus"] == node:
            network.loads.at[load_name, "p_set"] = 0.0


def set_node_up(network, node):
    """
    Reactivate a node's load (set to 0.02).
    """
    for load_name in network.loads.index:
        if network.loads.at[load_name, "bus"] == node:
            network.loads.at[load_name, "p_set"] = 0.02
