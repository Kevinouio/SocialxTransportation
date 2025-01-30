import traci
import random
from collections import defaultdict


import random
import traci

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


def main():
    # Path to your SUMO configuration file
    sumo_cfg = "osm.sumocfg"  # Replace with the path to your .sumocfg file

    # Start SUMO-GUI for visualization
    traci.start(["sumo-gui", "-c", sumo_cfg])

    # Define the dangerous street ID
    danger_street = "221597280#7"

    # Initialize a dictionary to track street crossing statistics
    street_statistics = defaultdict(int)

    try:
        step = 0
        while step < 1000:  # Simulate for 1000 steps
            traci.simulationStep()  # Advance the simulation

            # Get all active vehicles
            vehicle_ids = traci.vehicle.getIDList()

            for vehicle_id in vehicle_ids:
                # Record the current street (edge) the vehicle is on
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                street_statistics[current_edge] += 1

                # Apply rerouting logic before the vehicle enters the danger street
                reroute_vehicle_before_danger(vehicle_id, danger_street)

            step += 1
    finally:
        # Print statistics for streets that were crossed
        print("\nStreet Crossing Statistics:")
        for street, count in sorted(street_statistics.items(), key=lambda x: x[1], reverse=True):
            if count >= 0:  # Only print streets that were crossed
                print(f"{street}: {count} crossings")

        traci.close()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
