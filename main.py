import traci
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
from transformers import pipeline
import xml.etree.ElementTree as ET
import numpy as np

# Functions for Social Network Influence
def initialize_social_network(car_total):
    g = nx.complete_graph(car_total)
    # Algorithmic Bias model
    deez = op.AlgorithmicBiasModel(g)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", 0.32)
    config.add_model_parameter("gamma", 0)
    deez.set_initial_status(config)



    return deez


def get_street_names_from_network(network_file):
    tree = ET.parse(network_file)
    root = tree.getroot()
    street_names = []

    for edge in root.findall("edge"):
        if "name" in edge.attrib:
            street_names.append(edge.attrib["name"])

    return street_names

def map_street_names_to_edges(edge_ids):
    street_to_edges = {}
    for edge_id in edge_ids:
        street_name = traci.edge.getStreetName(edge_id)
        if street_name:  # Skip edges without street names
            if street_name not in street_to_edges:
                street_to_edges[street_name] = []
            street_to_edges[street_name].append(edge_id)
    return street_to_edges
def set_danger_levels_for_street(danger_street, danger_level, street_to_edges, danger_levels):
    if danger_street in street_to_edges:
        for edge_id in street_to_edges[danger_street]:
            danger_levels[edge_id] = danger_level  # Apply danger level to each edge

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


# Functions for LLM-based Rumor Evaluation
def evaluate_rumor_with_llm(rumor, street_names):
    sentiment_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    classification_pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

    sentiment = sentiment_pipe(rumor)[0]["label"]
    prompt = f"Does this rumor relate to any street listed below? {street_names}"
    street_classification = classification_pipe(prompt, max_length=1600)

    return sentiment, street_classification


# Dynamic Pathing Function
def reroute_vehicle(vehicle_id, danger_levels):
    current_edge = traci.vehicle.getRoadID(vehicle_id)
    outgoing_edges = traci.edge.getOutgoing(current_edge)
    next_edges = [edge[1].getID() for edge in outgoing_edges]

    best_edge = None
    best_score = float("inf")

    for edge in next_edges:
        occupancy = traci.edge.getLastStepOccupancy(edge)
        mean_speed = traci.edge.getLastStepMeanSpeed(edge)
        congestion_score = occupancy / max(mean_speed, 0.1)

        # Adjust the score based on danger level from social network
        danger_level = danger_levels.get(edge, 0)
        adjusted_score = congestion_score + danger_level

        if adjusted_score < best_score:
            best_edge = edge
            best_score = adjusted_score

    if best_edge:
        route_to_best_edge = traci.simulation.findRoute(current_edge, best_edge).edges
        traci.vehicle.setRoute(vehicle_id, route_to_best_edge)


# Main Simulation
def main():
    # Path to the .rou.xml file
    route_file = "osm.rou.xml"

    # Dynamically count vehicles in the route file
    car_total = count_vehicles_in_route_file(route_file)
    print(f"Total number of vehicles in the route file: {car_total}")

    social_model = initialize_social_network(car_total)
    edge_ids = traci.edge.getIDList()
    street_to_edges = map_street_names_to_edges(edge_ids)
    print(street_to_edges)
    danger_levels = {}  # Dictionary to store danger levels for edges

    # Parse and print street names from the network file
    network_file = "osm.net.xml"  # Path to your network file
    street_names = get_street_names_from_network(network_file)
    print("Street Names in the Network:", street_to_edges.keys())

    traci.start(["sumo-gui", "-n", network_file, "-r", route_file])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if rumor.lower() != "skip":
            sentiment, street_info = evaluate_rumor_with_llm(rumor, list(street_to_edges.keys()))
            print(f"Sentiment: {sentiment}, Street Info: {street_info}")

            # Mark the identified street as dangerous
            dangerous_street = "Harold Street"  # Example: Extracted from street_info
            danger_level = 1  # Example danger level
            set_danger_levels_for_street(dangerous_street, danger_level, street_to_edges, danger_levels)

            # Reroute vehicles based on updated danger levels
        for vehicle_id in traci.vehicle.getIDList():
            reroute_vehicle(vehicle_id, danger_levels)

    traci.close()



if __name__ == "__main__":
    main()
