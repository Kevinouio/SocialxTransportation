import traci
import random
from collections import defaultdict


def reroute_vehicle_before_danger(vehicle_id, danger_street):
    """
    Reroutes the vehicle if the next edge in its route is a dangerous street.
    Args:
        vehicle_id (str): The ID of the vehicle to be rerouted.
        danger_street (str): The ID of the street (edge) to be avoided.
    """
    # Get the current route and index
    route = traci.vehicle.getRoute(vehicle_id)
    route_index = traci.vehicle.getRouteIndex(vehicle_id)

    # Check if the danger street is in the next part of the route
    if route_index + 1 < len(route) and route[route_index + 1] == danger_street:
        # Get the current edge of the vehicle
        current_edge = traci.vehicle.getRoadID(vehicle_id)

        # Get all edges in the network
        all_edges = traci.edge.getIDList()

        # Find potential outgoing edges
        next_edges = [
            edge for edge in all_edges
            if current_edge in traci.simulation.findRoute(current_edge, edge).edges
        ]

        # Filter out routes that include the danger street
        safe_edges = []
        for edge in next_edges:
            route_to_edge = traci.simulation.findRoute(current_edge, edge).edges
            if danger_street not in route_to_edge:
                safe_edges.append(edge)

        # If there are safe edges, pick one randomly
        if safe_edges:
            new_edge = random.choice(safe_edges)
            route_to_new_edge = traci.simulation.findRoute(current_edge, new_edge).edges
            traci.vehicle.setRoute(vehicle_id, route_to_new_edge)
            print(f"Vehicle {vehicle_id} rerouted to avoid {danger_street}. New route: {route_to_new_edge}")
        else:
            print(f"No safe routes found for vehicle {vehicle_id} to avoid {danger_street}.")
    else:
        print(f"Vehicle {vehicle_id} is not approaching the danger street {danger_street}, no rerouting needed.")


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
