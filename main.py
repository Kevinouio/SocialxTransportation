import traci
import random
import time
import numpy as np
import pandas as pd

# Traffic simulation utilities
from network_utils import get_edge_to_street_mapping, get_street_to_edges_mapping, count_vehicles_in_route_file
from csv_utils import update_street_statistics_csv

# Power network simulation utilities
from powerNetworkGen import get_traffic_lights_from_sumo, get_road_edges_from_sumo, create_power_network, \
    add_buildings_from_poly
from powerNetwork import (
    run_power_flow,
    visualize_network_state,
    initialize_power_flow_log_df,
    append_power_flow_log_df,
    save_power_flow_log_df,
    simulate_local_partition_failure,
    set_node_down,
    set_node_up,
    add_ev_charging_station
)


def main():
    # -------------------------------
    # Traffic Simulation Setup
    # -------------------------------
    traffic_network_file = "osm.net.xml"
    route_file = "osm.rou.xml"

    car_total = count_vehicles_in_route_file(route_file)
    print(f"Total number of vehicles: {car_total}")

    edge_to_street, street_names, _ = get_edge_to_street_mapping(traffic_network_file)
    street_to_edges = get_street_to_edges_mapping(osm_file=traffic_network_file)
    street_crossings = {edge: 0 for edge in edge_to_street.keys()}

    traci.start(["sumo-gui", "-n", traffic_network_file, "-r", route_file])

    # -------------------------------
    # Power Network Simulation Setup
    # -------------------------------
    # Create base power network
    traffic_light_nodes = get_traffic_lights_from_sumo(traffic_network_file)
    road_edges = get_road_edges_from_sumo(traffic_network_file)
    power_network, sumo_to_label, label_to_sumo = create_power_network(traffic_light_nodes, road_edges, feeders=3)

    # Add buildings from poly file
    add_buildings_from_poly(power_network, "osm.poly.xml")

    # Add EV charging station with realistic load profile
    time_array = np.linspace(0, 2 * np.pi, 24)
    ev_load_profile_mw = 10 + 70 * np.cos(time_array - np.pi / 2) ** 2  # 10-50MW daily variation
    ev_load_profile = pd.Series(ev_load_profile_mw * 1000, index=power_network.snapshots)

    n30_coords = (power_network.buses.at["N30", 'x'], power_network.buses.at["N30", 'y'])
    add_ev_charging_station(
        network=power_network,
        station_name="EVStation1",
        connect_to_bus="N30",
        ev_load_profile_kw=ev_load_profile,
        bus_coords=(n30_coords[0] + 2, n30_coords[1] + 2),
        line_rating_mva=35
    )

    # Initialize power flow logging
    power_log_df = initialize_power_flow_log_df(power_network)
    current_power_step = 0
    power_down_nodes = set()

    # -------------------------------
    # Main Simulation Loop
    # -------------------------------
    tick_counter = -1

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            tick_counter += 1

            # Update traffic counts
            for edge in street_crossings.keys():
                street_crossings[edge] += traci.edge.getLastStepVehicleNumber(edge)

            # Update power network every 10 traffic steps
            if tick_counter % 10 == 0:
                current_power_step += 1

                # Automated outage selection
                choice = random.choices(["1", "2", "3"], weights=[5, 3, 2], k=1)[0]

                if choice == "2":
                    candidates = [b for b in power_network.buses.index
                                  if b not in ["MainPowerGrid", "LocalSubstation"] and b not in power_down_nodes]
                    if candidates:
                        fail_node = random.choice(candidates)
                        power_down_nodes.add(fail_node)
                        set_node_down(power_network, fail_node)

                if choice == "3":
                    simulate_local_partition_failure(power_network, power_down_nodes, depth=2)

                # Random recovery
                if random.random() < 0.1 and power_down_nodes:
                    recov = random.choice(list(power_down_nodes))
                    power_down_nodes.remove(recov)
                    set_node_up(power_network, recov)

                # Run power flow and log results
                run_power_flow(power_network)
                power_log_df = append_power_flow_log_df(power_network, power_log_df, current_power_step)
                visualize_network_state(power_network, power_down_nodes, time_step=current_power_step)
                time.sleep(0.1)

    finally:
        # -------------------------------
        # Simulation Cleanup
        # -------------------------------
        # Save power logs
        save_power_flow_log_df(power_log_df, "combined_power_log.csv")

        # Save traffic statistics
        grouped_stats = {}
        for edge, count in street_crossings.items():
            street_name = edge_to_street.get(edge, "Unknown")
            grouped_stats[f"{street_name} ({edge})"] = count
        update_street_statistics_csv(grouped_stats, "No Rumor Injected")

        traci.close()
        print("Co-simulation completed successfully")


if __name__ == "__main__":
    main()
