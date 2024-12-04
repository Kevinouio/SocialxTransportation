import traci

from assign.costFunctionChecker import generateWeights

import socialNetwork
from socialNetwork import createWeights

# Functions for rerouting

def reroute_vehicle(vehicle_id):
    # Get current edge
    current_edge = traci.vehicle.getRoadID(vehicle_id)

    # Define possible next edges from the current edge
    outgoing_edges = traci.edge.getOutgoing(current_edge)
    next_edges = [edge[1].getID() for edge in outgoing_edges]

    # Analyze traffic conditions on each next edge
    best_edge = None
    best_score = float("inf")

    for edge in next_edges:
        # Use occupancy and speed as metrics
        occupancy = traci.edge.getLastStepOccupancy(edge)
        mean_speed = traci.edge.getLastStepMeanSpeed(edge)

        # Calculate a "congestion score"
        congestion_score = occupancy / max(mean_speed, 0.1)  # Avoid division by zero

        # Select the edge with the lowest congestion score
        if congestion_score < best_score:
            best_edge = edge
            best_score = congestion_score

    # If a better edge is found, reroute the vehicle
    if best_edge:
        route_to_best_edge = traci.simulation.findRoute(current_edge, best_edge).edges
        traci.vehicle.setRoute(vehicle_id, route_to_best_edge)


def main():
    traci.start(["sumo-gui", "-n", "your_network.net.xml", "-r", "your_routes.rou.xml"])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()  # Advance the simulation

        # Check and reroute each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            reroute_vehicle(vehicle_id)

    traci.close()

# Note Car names are t_0, t_1, ... etc.
# Initial variables needed for the cars
numOfCars = 20
weightsPerIteration = []
totalTime = 20000     # Seconds

# Gets a list of the nodes of the cars that are within the

model,iterations, CarWeights = createWeights(100,100)

