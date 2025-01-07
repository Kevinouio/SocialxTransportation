import xml.etree.ElementTree as ET

def get_edge_to_street_mapping(osm_file="osm.net.xml"):
    # Parse the XML file
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Initialize dictionaries to store mapping
    edge_to_street = {}
    street_names = set()  # Use a set to avoid duplicate street names

    # Iterate over all <edge> elements in the XML
    for edge in root.findall("edge"):
        edge_id = edge.get("id")  # Get the edge ID
        street_name = edge.get("name")  # Get the name attribute

        if street_name:  # Only consider edges with a valid name
            edge_to_street[edge_id] = street_name
            street_names.add(street_name)

    # Convert street_names back to a sorted list
    return edge_to_street, sorted(street_names)

if __name__ == "__main__":
    # Test the function
    edge_to_street, street_names = get_edge_to_street_mapping("osm.net.xml")
    print(f"Edge to Street Mapping: {edge_to_street}")
    print(f"List of Street Names: {street_names}")
