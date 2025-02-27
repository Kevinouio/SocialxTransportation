import xml.etree.ElementTree as ET

def get_street_names_from_network(network_file):
    tree = ET.parse(network_file)
    root = tree.getroot()
    street_names = [edge.attrib["name"] for edge in root.findall("edge") if "name" in edge.attrib]
    return street_names

def get_edge_to_street_mapping(osm_file="osm.net.xml"):
    tree = ET.parse(osm_file)
    root = tree.getroot()
    edge_to_street = {}
    street_names = set()
    street_to_danger_level = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        street_name = edge.get("name")
        if street_name:
            edge_to_street[edge_id] = street_name
            street_names.add(street_name)
    for street in street_names:
        street_to_danger_level[street] = 0
    return edge_to_street, sorted(street_names), street_to_danger_level

def get_street_to_edges_mapping(osm_file="osm.net.xml"):
    tree = ET.parse(osm_file)
    root = tree.getroot()
    street_to_edges = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        street_name = edge.get("name")
        if street_name:
            street_to_edges.setdefault(street_name, []).append(edge_id)
    return street_to_edges

def count_vehicles_in_route_file(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot()
    vehicle_count = len(root.findall("vehicle"))
    return vehicle_count
