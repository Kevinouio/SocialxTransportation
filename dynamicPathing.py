import traci
import random

def reroute_vehicle_with_multiple_rumors(vehicle_id, social_models, vehicle_to_node):
    node_id = vehicle_to_node.get(vehicle_id)
    if node_id is None:
        print(f"Vehicle {vehicle_id} has no assigned node.")
        return

    dangerous_edges = set()
    for social_model in social_models:
        if social_model.status.get(node_id, 0) == 1:
            dangerous_edges.update(social_model.related_edges)

    if not dangerous_edges:
        return

    route = traci.vehicle.getRoute(vehicle_id)
    route_index = traci.vehicle.getRouteIndex(vehicle_id)
    if route_index + 1 < len(route) and route[route_index + 1] in dangerous_edges:
        current_edge = traci.vehicle.getRoadID(vehicle_id)
        all_edges = traci.edge.getIDList()
        safe_edges = [
            edge for edge in all_edges
            if current_edge in traci.simulation.findRoute(current_edge, edge).edges
            and all(dangerous_edge not in traci.simulation.findRoute(current_edge, edge).edges for dangerous_edge in dangerous_edges)
        ]
        for new_edge in safe_edges:
            route_result = traci.simulation.findRoute(current_edge, new_edge)
            if route_result.edges and len(route_result.edges) > 1:
                new_route = route_result.edges
                traci.vehicle.setRoute(vehicle_id, new_route)
                traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                print(f"Vehicle {vehicle_id} rerouted to avoid {dangerous_edges}. New route: {new_route}")
                return
        print(f"No safe alternative routes found for vehicle {vehicle_id} to avoid {dangerous_edges}.")
