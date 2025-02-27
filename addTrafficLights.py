import xml.etree.ElementTree as ET

def add_tl_logic_to_network(input_net, output_net):
    """
    Reads the input SUMO network file, adds traffic light logic to eligible junctions,
    and writes out a new network file with traffic lights.

    Args:
        input_net (str): Path to the input network file (e.g., 'osm.net.xml').
        output_net (str): Path for the modified network file (e.g., 'osm_with_tls.net.xml').
    """
    # Parse the network XML file
    tree = ET.parse(input_net)
    root = tree.getroot()

    # Iterate over all junction elements
    # Typically, junctions with types "priority" or "unregulated" might not have traffic lights.
    # We'll add a tlLogic element for each junction missing one.
    for junction in root.findall("junction"):
        jtype = junction.get("type")
        j_id = junction.get("id")
        if jtype in ["priority", "unregulated"]:
            # Check if a traffic light logic for this junction already exists
            existing = root.find(f".//tlLogic[@id='{j_id}']")
            if existing is None:
                print(f"Adding traffic light at junction {j_id}")
                # Create a new tlLogic element with a simple fixed-time program
                tl = ET.Element("tlLogic", attrib={
                    "id": j_id,
                    "type": "static",
                    "programID": "0",
                    "offset": "0"
                })
                # Define phases (example: long red, short green) — adjust as needed
                # Here, "r" indicates red, "G" indicates green.
                ET.SubElement(tl, "phase", attrib={"duration": "40", "state": "rrGG"})  # Mostly red
                ET.SubElement(tl, "phase", attrib={"duration": "10", "state": "GGrr"})  # Green for a short period
                # Append the new tlLogic element to the root
                root.append(tl)

    # Write out the modified network file
    tree.write(output_net)
    print(f"Modified network saved as {output_net}")

if __name__ == "__main__":
    # Modify these file names as needed
    input_network = "osm.net.xml"
    output_network = "osm_with_tls.net.xml"
    add_tl_logic_to_network(input_network, output_network)
