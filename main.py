import traci
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
from transformers import pipeline
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# Functions for Social Network Influence
def initialize_social_network(car_total):
    g = nx.complete_graph(car_total)
    # Algorithmic Bias model
    model = op.AlgorithmicBiasModel(g)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", 0.32)
    config.add_model_parameter("gamma", 0)
    model.set_initial_status(config)

    return model


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


# Visualize the social network and rumor propagation
def visualize_social_network(car_total, statuses=None):
    _, g = initialize_social_network(car_total)

    # Node colors based on statuses
    node_colors = []
    if statuses:
        for node in g.nodes():
            # Color code: 1 = Believes rumor, 0 = Doesn't believe rumor
            node_colors.append("red" if statuses.get(node, 0) >= 0.5 else "blue")
    else:
        node_colors = ["blue"] * len(g.nodes())  # Default color for all nodes

    # Plot the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g)  # Position for a spring layout
    nx.draw(g, pos, node_color=node_colors, with_labels=True, node_size=500, edge_color="gray")
    plt.title("Social Network Visualization with Rumor Propagation")
    plt.show()


# Functions for LLM-based Rumor Evaluation
def evaluate_rumor_with_llm(rumor, street_names):
    from transformers import pipeline

    # Initialize pipelines
    sentiment_pipe = pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True
    )
    classification_pipe = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )

    # Get sentiment analysis results
    sentiment_scores = sentiment_pipe(rumor)[0]  # Returns a list of dictionaries with scores for each label
    sentiment_results = {entry["label"]: entry["score"] for entry in sentiment_scores}

    # Determine overall sentiment
    if sentiment_results.get("negative", 0) > 0.40:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    # Street classification
    classification_result = classification_pipe(rumor, candidate_labels=street_names)
    relevant_streets = [
        street for street, score in zip(classification_result["labels"], classification_result["scores"]) if score > 0.5
    ]

    # Print results for debugging
    print(f"Relevant Streets: {relevant_streets}")
    print(f"Sentiment Results: {sentiment_results}")
    print(f"Overall Sentiment: {overall_sentiment}")

    return overall_sentiment, relevant_streets


# Dynamic Pathing Function
def reroute_vehicle(vehicle_id, danger_levels, social_network_status):
    """
    Reroutes a vehicle based on its awareness of a rumor and the danger levels of streets.

    Args:
        vehicle_id (str): The ID of the vehicle to be rerouted.
        danger_levels (dict): Dictionary mapping street names to danger levels.
        social_network_status (dict): Dictionary mapping vehicle IDs to rumor awareness levels (0 to 1).
    """
    # Check if the vehicle "heard the rumor"
    if social_network_status.get(vehicle_id, 0) < 0.5:
        # If the vehicle has not "heard the rumor," do not reroute
        print("Did Not Hear Rumor")
        return

    # Get the current edge of the vehicle
    current_edge = traci.vehicle.getRoadID(vehicle_id)

    # Get outgoing edges from the current edge
    outgoing_edges = traci.edge.getOutgoing(current_edge)
    next_edges = [edge[1].getID() for edge in outgoing_edges]

    # Evaluate the best edge based on danger levels
    best_edge = None
    best_score = float("inf")

    for edge in next_edges:
        # Get the danger level for the edge
        danger_level = danger_levels.get(edge, 0)

        # Skip edges with non-zero danger levels (avoid dangerous streets)
        if danger_level > 0:
            continue

        # Default score for now (can be expanded later with congestion factors)
        adjusted_score = danger_level

        if adjusted_score < best_score:
            best_edge = edge
            best_score = adjusted_score

    # If a safe edge is found, reroute the vehicle
    if best_edge:
        route_to_best_edge = traci.simulation.findRoute(current_edge, best_edge).edges
        traci.vehicle.setRoute(vehicle_id, route_to_best_edge)


# Main Simulation
def main():
    # Paths to the .net.xml and .rou.xml files
    network_file = "osm.net.xml"  # Path to your network file
    route_file = "osm.rou.xml"  # Path to your route file

    # Dynamically count vehicles in the route file
    car_total = count_vehicles_in_route_file(route_file)
    print(f"Total number of vehicles in the route file: {car_total}")

    # Initialize the social network model for rumor propagation
    social_model = initialize_social_network(car_total)

    # Parse edge-to-street mapping and initialize danger levels
    edge_to_street, street_names, danger_levels = get_edge_to_street_mapping(network_file)

    # Start the SUMO simulation
    traci.start(["sumo-gui", "-n", network_file, "-r", route_file])

    # Initialize tick counter
    tick_counter = -1
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # Advance simulation step
            traci.simulationStep()
            tick_counter += 1
            if tick_counter == 0:
                # Prompt for a rumor input
                rumor = input("Enter a rumor about a street (or type 'skip'): ")
                if rumor.lower() == "skip":
                    continue

                # Evaluate the rumor using LLMs
                sentiment, street_info = evaluate_rumor_with_llm(rumor, street_names)
                print(f"Sentiment: {sentiment}")
                print(f"Street Info: {street_info}")

                # Extract relevant streets from the street_info
                # Since street_info is a list of strings
                relevant_streets = [info.strip() for info in street_info if info.strip() in street_names]

                print(f"Relevant Streets Identified: {relevant_streets}")

                # Update danger levels for relevant streets
                for street in relevant_streets:
                    danger_levels[street] = 1  # Mark relevant streets as dangerous
                    print(f"Updated danger level for {street} to 1")
                # Example social network status for vehicles (REMEMBER TO CHANGE THIS BACK)
                for i in range(20):
                    social_network_status = propagate_rumor(social_model, rumor)
                visualize_social_network(car_total, social_network_status)

            # Update vehicle routing
            for vehicle_id in traci.vehicle.getIDList():
                reroute_vehicle(vehicle_id, danger_levels, social_network_status)


    finally:
        # Close the SUMO simulation
        traci.close()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
