import traci
import random
import time

# Traffic simulation utilities
from network_utils import get_edge_to_street_mapping, get_street_to_edges_mapping, count_vehicles_in_route_file
from csv_utils import update_street_statistics_csv

# Power network simulation utilities
from powerNetworkGen import get_traffic_lights_from_sumo, get_road_edges_from_sumo, create_power_network
from powerNetwork import (
    run_power_flow,
    visualize_network_state,
    initialize_csv_log,
    append_csv_column,
    simulate_local_partition_failure,
    set_node_down,
    set_node_up
)


def main():
    # -------------------------------
    # Traffic Simulation Setup
    # -------------------------------
    traffic_network_file = "new_network.net.xml"  # SUMO network for traffic simulation
    route_file = "osm.rou.xml"
    poly_file = "osm.poly.xml"  # background polygons, if needed

    car_total = count_vehicles_in_route_file(route_file)
    print(f"Total number of vehicles: {car_total}")

    edge_to_street, street_names, _ = get_edge_to_street_mapping(traffic_network_file)
    street_to_edges = get_street_to_edges_mapping(osm_file=traffic_network_file)
    street_crossings = {edge: 0 for edge in edge_to_street.keys()}

    # Start the SUMO traffic simulation
    traci.start(["sumo-gui", "-n", traffic_network_file, "-r", route_file, "-a", poly_file])

    # -------------------------------
    # Power Network Simulation Setup
    # -------------------------------
    power_network_file = "osm.net.xml"  # Power network file (can be same as or different from the traffic network)
    traffic_light_nodes = get_traffic_lights_from_sumo(power_network_file)
    road_edges = get_road_edges_from_sumo(power_network_file)
    power_network, sumo_to_label, label_to_sumo = create_power_network(traffic_light_nodes, road_edges, feeders=3)

    # Prepare CSV logging for the power network
    voltages = initialize_csv_log(power_network, "node_stats.csv")
    power_down_nodes = set()
    current_power_step = 0

    # -------------------------------
    # Main Simulation Loop
    # -------------------------------
    tick_counter = -1

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            tick_counter += 1

            # ----- Traffic Simulation Updates -----
            for edge in street_crossings.keys():
                street_crossings[edge] += traci.edge.getLastStepVehicleNumber(edge)

            # ----- Power Network Simulation Updates -----
            # Update power network every 10 ticks
            if tick_counter % 10 == 0:
                current_power_step += 1

                # Introduce a random node failure (10% chance)
                if random.random() < 0.1:
                    candidates = [b for b in power_network.buses.index
                                  if b not in ["MainPowerGrid", "LocalSubstation"] and b not in power_down_nodes]
                    if candidates:
                        fail_node = random.choice(candidates)
                        print(f"Power Network: Failing node {fail_node}!")
                        power_down_nodes.add(fail_node)
                        set_node_down(power_network, fail_node)

                # Random node recovery (10% chance)
                if random.random() < 0.1 and power_down_nodes:
                    recov = random.choice(list(power_down_nodes))
                    print(f"Power Network: Recovering node {recov}!")
                    power_down_nodes.remove(recov)
                    set_node_up(power_network, recov)

                # Simulate a local partition failure (40% chance)
                if random.random() < 0.4:
                    simulate_local_partition_failure(power_network, power_down_nodes, depth=2)

                run_power_flow(power_network)
                append_csv_column(power_network, voltages, current_power_step, "node_stats.csv")
                visualize_network_state(power_network, power_down_nodes, time_step=current_power_step)
                time.sleep(0.1)

    finally:
        # ----- Traffic Simulation Cleanup -----
        grouped_street_crossings = {}
        for edge, count in street_crossings.items():
            street_name = edge_to_street.get(edge, "Unknown Street")
            base_edge = edge.lstrip("-")
            grouped_street_crossings.setdefault(base_edge, []).append((edge, count, street_name))
        print("\nTraffic Street Crossing Statistics (Grouped):")
        for base_edge, edges in grouped_street_crossings.items():
            edges.sort(key=lambda x: x[0])
            for edge, count, street_name in edges:
                print(f"{street_name} ({edge}): {count} crossings")
        street_stats = {}
        for base_edge, edges in grouped_street_crossings.items():
            for edge, count, street_name in edges:
                street_stats[f"{street_name} ({edge})"] = count

        # Since no rumors are injected, we log a placeholder for the rumor column
        update_street_statistics_csv(street_stats, "No Rumor Injected")
        print("Traffic street statistics updated.")

        traci.close()
        print("Traffic simulation ended.")
        print("Power network simulation ended.")


if __name__ == "__main__":
    main()
