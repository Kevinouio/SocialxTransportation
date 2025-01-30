import traci
import networkx as nx
from transformers import pipeline
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
import socialNetwork as sn



def get_street_names_from_network(network_file):
    tree = ET.parse(network_file)
    root = tree.getroot()
    street_names = []

    for edge in root.findall("edge"):
        if "name" in edge.attrib:
            street_names.append(edge.attrib["name"])

    return street_names


def get_edge_to_street_mapping(osm_file="osm.net.xml"):
    # Parse the XML file
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Initialize dictionaries to store mapping
    edge_to_street = {}
    street_names = set()  # Use a set to avoid duplicate street names
    street_to_danger_level = {}

    # Iterate over all <edge> elements in the XML
    for edge in root.findall("edge"):
        edge_id = edge.get("id")  # Get the edge ID
        street_name = edge.get("name")  # Get the name attribute

        if street_name:  # Only consider edges with a valid name
            edge_to_street[edge_id] = street_name
            street_names.add(street_name)

    # Initialize danger levels for each street
    for street_name in street_names:
        street_to_danger_level[street_name] = 0  # Default danger level is 0

    # Convert street_names back to a sorted list
    return edge_to_street, sorted(street_names), street_to_danger_level

def get_street_to_edges_mapping(osm_file="osm.net.xml"):
    """
    Parses a SUMO network file and maps each street name to the list of edge IDs it contains.

    Args:
        osm_file (str): Path to the SUMO network file (osm.net.xml).

    Returns:
        dict: A dictionary mapping street names to lists of edge IDs.
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()

    street_to_edges = {}

    for edge in root.findall("edge"):
        edge_id = edge.get("id")  # Get the edge ID
        street_name = edge.get("name")  # Get the street name

        if street_name:  # Only include edges that have a valid street name
            if street_name not in street_to_edges:
                street_to_edges[street_name] = []
            street_to_edges[street_name].append(edge_id)

    return street_to_edges

def count_vehicles_in_route_file(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot()
    vehicle_count = len(root.findall("vehicle"))
    return vehicle_count


def propagate_rumor(model, rumor):
    # Simulate rumor propagation
    iterations = model.iteration_bunch(1)
    statuses = iterations[-1]["status"]  # Get the latest opinions
    return statuses


def evaluate_rumor_with_llm(rumor, street_names):
    """
    Evaluates a rumor prompt with LLMs for sentiment and street classification.
    Args:
        rumor (str): The rumor prompt to evaluate.
        street_names (list): A list of street names for classification.
    Returns:
        tuple: Overall sentiment and relevant streets.
    """
    # Initialize pipelines
    sentiment_pipe = pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True
    )
    classification_pipe = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )

    # Get sentiment analysis results
    sentiment_scores = sentiment_pipe(rumor)[0]
    sentiment_results = {entry["label"]: entry["score"] for entry in sentiment_scores}

    # Determine overall sentiment
    if sentiment_results.get("negative", 0) > 0.35:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    # Street classification
    classification_result = classification_pipe(rumor, candidate_labels=street_names)
    relevant_streets = [
        street for street, score in zip(classification_result["labels"], classification_result["scores"]) if score > 0.5
    ]

    print(f"Relevant Streets: {relevant_streets}")
    print(f"Sentiment Results: {sentiment_results}")
    print(f"Overall Sentiment: {overall_sentiment}")

    return overall_sentiment, relevant_streets

def generate_prompts_based_on_cars(cartotal, street_names):
    """
    Generates prompts based on the total number of cars.
    Args:
        cartotal (int): The total number of cars in the simulation.
        street_names (list): A list of street names.
    Returns:
        list: A list of generated prompts.
    """
    num_prompts = max(1, cartotal // 3)  # Ensure at least one prompt is generated
    prompts = []

    for _ in range(num_prompts):
        event_type = random.choice(["active shooter", "traffic jam"])
        street = random.choice(street_names)
        prompts.append(f"There is a {event_type} at {street}.")

    return prompts




def reroute_vehicle_with_multiple_rumors(vehicle_id, social_models, vehicle_to_node):
    """
    Reroutes a vehicle if its corresponding node in the social network is infected in any of the active rumors.
    Args:
        vehicle_id (str): The ID of the vehicle to be rerouted.
        social_models (list): List of SocialNetwork objects, each containing an associated dangerous street edge.
        vehicle_to_node (dict): Mapping of vehicle IDs to their corresponding social network nodes.
    """
    # Get the node associated with this vehicle
    node_id = vehicle_to_node.get(vehicle_id)
    if node_id is None:
        print(f"Vehicle {vehicle_id} has no assigned node.")
        return

    # Collect all dangerous edges from infected rumors
    dangerous_edges = set()
    for social_model in social_models:
        if social_model.status.get(node_id, 0) == 1:  # Node is infected in this rumor
            dangerous_edges.add(social_model.related_edge)

    # Skip rerouting if there are no active dangerous edges for this vehicle
    if not dangerous_edges:
        return

    # Get the vehicle’s current route and index
    route = traci.vehicle.getRoute(vehicle_id)
    route_index = traci.vehicle.getRouteIndex(vehicle_id)

    # Check if the upcoming edge is dangerous
    if route_index + 1 < len(route) and route[route_index + 1] in dangerous_edges:
        current_edge = traci.vehicle.getRoadID(vehicle_id)

        # Get all edges in the network
        all_edges = traci.edge.getIDList()

        # Find alternative outgoing edges that do not contain a dangerous edge
        safe_edges = [
            edge for edge in all_edges
            if current_edge in traci.simulation.findRoute(current_edge, edge).edges
            and all(dangerous_edge not in traci.simulation.findRoute(current_edge, edge).edges for dangerous_edge in dangerous_edges)
        ]

        # If there are safe edges, pick one randomly and reroute
        if safe_edges:
            new_edge = random.choice(safe_edges)
            new_route = traci.simulation.findRoute(current_edge, new_edge).edges
            traci.vehicle.setRoute(vehicle_id, new_route)
            print(f"Vehicle {vehicle_id} rerouted to avoid {dangerous_edges}. New route: {new_route}")
        else:
            print(f"No safe alternative routes found for vehicle {vehicle_id} to avoid {dangerous_edges}.")


#Main simulation
def main():
    # Paths to the .net.xml and .rou.xml files
    network_file = "osm.net.xml"  # Path to your network file
    route_file = "osm.rou.xml"  # Path to your route file

    # Dynamically count vehicles in the route file
    car_total = count_vehicles_in_route_file(route_file)
    print(f"Total number of vehicles in the route file: {car_total}")

    # Parse edge-to-street mapping and initialize danger levels
    edge_to_street, street_names, danger_levels = get_edge_to_street_mapping(network_file)
    street_to_edges = get_street_to_edges_mapping(osm_file=network_file)

    # Initialize dictionary to track street crossings
    street_crossings = {edge: 0 for edge in edge_to_street.keys()}
    print(street_to_edges)

    # Mapping of vehicles to social network nodes
    vehicle_to_node = {}

    # Generates the prompts that will be injected into the simulation
    prompts = generate_prompts_based_on_cars(car_total, street_names)

    # Start the SUMO simulation
    traci.start(["sumo-gui", "-n", network_file, "-r", route_file])

    # Initialize variables
    tick_counter = -1
    rumor_list = []  # List to store rumors
    social_networks = []  # List to store social network objects

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # Advance simulation step
            traci.simulationStep()
            tick_counter += 1

            # Update street crossing counts
            for edge in street_crossings.keys():
                street_crossings[edge] += traci.edge.getLastStepVehicleNumber(edge)

                # Track the order of car generation and assign to social network nodes
                departed_vehicles = traci.simulation.getDepartedIDList()
                for vehicle_id in departed_vehicles:
                    # Map vehicle to a node based on order (modulo the total number of nodes)
                    assigned_node = len(vehicle_to_node) % car_total
                    vehicle_to_node[vehicle_id] = assigned_node
                    # Log the assignment
                    print(f"Vehicle ID: {vehicle_id} -> Assigned Node ID: {assigned_node}")


            # Inject rumors every 50 seconds with 20% probability
            if tick_counter > 0 and tick_counter % 50 == 0:
                if random.random() <= 0.2:


                    # Generate a rumor input and add to the list
                    #rumor = input("Enter a rumor about a street (or type 'skip'): ")
                    rumor = random.choice(prompts)
                    sentiment, street_name = evaluate_rumor_with_llm(rumor, street_names)
                    streetID = random.choice(street_to_edges[street_name[0]])
                    if sentiment=="negative":
                        # Generate a new social network model for the rumor
                        social_network = sn.SocialNetwork(node_count=car_total, recovery_delay=10, rumor_count=len(rumor_list) + 1, related_edge=streetID)
                        rumor_list.append(rumor)
                        social_networks.append(social_network)
                        print(f"Rumor {len(rumor_list)} added: {rumor}")

            # Run a timestep for each social network every 50 seconds
            if tick_counter > 0 and tick_counter % 100 == 0:
                for idx, social_network in enumerate(social_networks):
                    print(f"Running timestep for Rumor {idx + 1}")
                    social_network.run_time_step()
                    social_network.visualize()

            # Update vehicle routing
            for vehicle_id in traci.vehicle.getIDList():
                reroute_vehicle_with_multiple_rumors(vehicle_id, social_models=social_networks, vehicle_to_node=vehicle_to_node)


    finally:
        # Print street crossing statistics
        print("Street Crossing Statistics:")
        for edge, count in street_crossings.items():
            street_name = edge_to_street.get(edge, "Unknown Street")
            print(f"{street_name} ({edge}): {count} crossings")

        # Optionally, save statistics to a file
        with open("street_crossings.txt", "w") as f:
            for edge, count in street_crossings.items():
                street_name = edge_to_street.get(edge, "Unknown Street")
                f.write(f"{street_name} ({edge}): {count} crossings\n")

        # Close the SUMO simulation
        traci.close()
        print("Simulation ended.")




if __name__ == "__main__":
    main()
