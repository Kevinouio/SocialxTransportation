import traci
import random
import os
import socialNetwork as sn

from network_utils import get_edge_to_street_mapping, get_street_to_edges_mapping, count_vehicles_in_route_file
from LLMmodelRunner import evaluate_rumor_with_llm, generate_prompts_based_on_cars
from dynamicPathing import reroute_vehicle_with_multiple_rumors
from csv_utils import update_street_statistics_csv


def main():
    network_file = "osm.net.xml"
    route_file = "osm.rou.xml"
    # For background polygons from OSM Web Wizard:
    poly_file = "osm.poly.xml"

    car_total = count_vehicles_in_route_file(route_file)
    print(f"Total number of vehicles: {car_total}")

    edge_to_street, street_names, danger_levels = get_edge_to_street_mapping(network_file)
    street_to_edges = get_street_to_edges_mapping(osm_file=network_file)
    street_crossings = {edge: 0 for edge in edge_to_street.keys()}

    vehicle_to_node = {}
    prompted = generate_prompts_based_on_cars(car_total, street_names)
    prompts = [random.choice(prompted) for _ in range(2)]

    traci.start(["sumo-gui", "-n", network_file, "-r", route_file, "-a", poly_file])

    tick_counter = -1
    rumor_list = []
    social_networks = []
    dangerous_edges = []

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            tick_counter += 1

            for edge in street_crossings.keys():
                street_crossings[edge] += traci.edge.getLastStepVehicleNumber(edge)
                for vehicle_id in traci.simulation.getDepartedIDList():
                    if vehicle_id not in vehicle_to_node:
                        assigned_node = len(vehicle_to_node) % car_total
                        vehicle_to_node[vehicle_id] = assigned_node

            if tick_counter > 0 and tick_counter % 50 == 0 and prompts:
                rumor = random.choice(prompts)
                prompts.remove(rumor)
                sentiment, street_name = evaluate_rumor_with_llm(rumor, street_names)
                streetID = random.choice(street_to_edges[street_name[0]])
                negative_streetID = f"-{streetID}" if not streetID.startswith("-") else streetID.lstrip("-")
                edges_to_add = [streetID]
                if negative_streetID in street_to_edges.get(street_name[0], []):
                    edges_to_add.append(negative_streetID)
                dangerous_edges.extend(edges_to_add)
                if sentiment == "negative":
                    social_network = sn.SocialNetwork(node_count=car_total, recovery_delay=10,
                                                      rumor_count=len(rumor_list) + 1, related_edges=edges_to_add)
                    rumor_list.append(rumor)
                    social_networks.append(social_network)
                    print(f"Rumor {len(rumor_list)} added: {rumor}")

            if tick_counter > 0 and tick_counter % 75 == 0:
                for idx, social_network in enumerate(social_networks):
                    print(f"Running timestep for Rumor {idx + 1}")
                    social_network.run_time_step()
                    social_network.visualize()

            for vehicle_id in traci.vehicle.getIDList():
                reroute_vehicle_with_multiple_rumors(vehicle_id, social_models=social_networks,
                                                     vehicle_to_node=vehicle_to_node)
    finally:
        grouped_street_crossings = {}
        for edge, count in street_crossings.items():
            street_name = edge_to_street.get(edge, "Unknown Street")
            base_edge = edge.lstrip("-")
            grouped_street_crossings.setdefault(base_edge, []).append((edge, count, street_name))
        print("\nStreet Crossing Statistics (Grouped):")
        for base_edge, edges in grouped_street_crossings.items():
            edges.sort(key=lambda x: x[0])
            for edge, count, street_name in edges:
                print(f"{street_name} ({edge}): {count} crossings")
        street_stats = {}
        for base_edge, edges in grouped_street_crossings.items():
            for edge, count, street_name in edges:
                street_stats[f"{street_name} ({edge})"] = count
        rumor_street = str(dangerous_edges)
        update_street_statistics_csv(street_stats, rumor_street)
        print("Street statistics updated.")
        traci.close()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
