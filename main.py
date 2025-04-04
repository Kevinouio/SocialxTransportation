import traci
import random
import time
import numpy as np
import pandas as pd
import logging

# Suppress PyPSA logging
logging.getLogger('pypsa').setLevel(logging.ERROR)
logging.getLogger('pypsa.pf').setLevel(logging.CRITICAL)

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
    tl_ids = traci.trafficlight.getIDList()

    # -------------------------------
    # Power Network Simulation Setup (UPDATED)
    # -------------------------------
    traffic_light_nodes = get_traffic_lights_from_sumo(traffic_network_file)
    road_edges = get_road_edges_from_sumo(traffic_network_file)
    power_network, sumo_to_label, label_to_sumo, _ = create_power_network(traffic_light_nodes, road_edges, feeders=3)

    # Set up temporal parameters FIRST
    total_steps = 24
    power_network.set_snapshots(range(total_steps))

    # Now add buildings and EV stations with time-aware profiles
    add_buildings_from_poly(power_network, "osm.poly.xml")

    # Configure massive EV load to trigger voltage drops
    n30_coords = (power_network.buses.at["N30", "x"], power_network.buses.at["N30", "y"])
    time_array = np.linspace(0, 2 * np.pi, total_steps)
    ev_load_profile_mw = 500 + 1500 * np.cos(time_array - np.pi / 2) ** 2  # 500-2000MW peak

    add_ev_charging_station(
        power_network,
        station_name="EVStation1",
        connect_to_bus="N30",
        ev_load_profile_kw=pd.Series(ev_load_profile_mw * 1000, index=power_network.snapshots),
        bus_coords=(n30_coords[0] + 2, n30_coords[1] + 2),
        line_rating_mva=500,  # Increased capacity
        line_resistance_per_km=0.01,
        line_reactance_per_km=0.03
    )

    # Initialize tracking
    voltage_threshold = 0.8  # ANSI minimum voltage standard
    power_log_df = initialize_power_flow_log_df(power_network)
    current_power_step = 0
    power_down_nodes = {}  # {node: (failed_step, recovery_step)}
    tl_to_power_node = {tl_id: sumo_to_label[tl_id] for tl_id in tl_ids if tl_id in sumo_to_label}

    # -------------------------------
    # Main Simulation Loop (UPDATED)
    # -------------------------------
    tick_counter = 0
    power_update_interval = 10  # Update power network every 10 traffic steps
    failure_duration = 10  # Keep nodes down for 10 power steps (~2.4 hours)

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            tick_counter += 1

            # Update traffic counts
            for edge in street_crossings.keys():
                street_crossings[edge] += traci.edge.getLastStepVehicleNumber(edge)

            # Power network updates
            if tick_counter % power_update_interval == 0:
                current_power_step += 1

                try:
                    # Run power flow and check voltages
                    run_power_flow(power_network)
                    current_voltages = power_network.buses_t.v_mag_pu.iloc[current_power_step % 24]

                    # Detect voltage violations
                    new_failures = current_voltages[current_voltages < voltage_threshold].index.tolist()
                    for node in new_failures:
                        if node not in power_down_nodes and node not in ["MainPowerGrid", "LocalSubstation"]:
                            power_down_nodes[node] = (current_power_step, current_power_step + failure_duration)
                            set_node_down(power_network, node)
                            print(f"⚡ Node {node} failed due to low voltage {current_voltages[node]:.3f} pu")

                    # Handle recoveries
                    nodes_to_recover = [
                        node for node, (_, recovery) in power_down_nodes.items()
                        if current_power_step >= recovery
                    ]
                    for node in nodes_to_recover:
                        set_node_up(power_network, node)
                        del power_down_nodes[node]
                        print(f"🔌 Node {node} restored after {failure_duration} power steps")

                    # Update visualization and logs
                    power_log_df = append_power_flow_log_df(power_network, power_log_df, current_power_step)
                    visualize_network_state(power_network, power_down_nodes.keys(), time_step=current_power_step)

                except Exception as e:
                    print(f"Power flow error: {str(e)[:50]}")

                # Update traffic light states based on power nodes
                for tl_id, power_node in tl_to_power_node.items():
                    current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                    if power_node in power_down_nodes:
                        if 'r' not in current_state.lower():
                            traci.trafficlight.setRedYellowGreenState(tl_id, "rrrr")
                    else:
                        if current_state != "GGgg":
                            traci.trafficlight.setRedYellowGreenState(tl_id, "GGgg")

            time.sleep(0.05)

    except Exception as e:
        print(f"Simulation ended: {str(e)}")
    finally:
        save_power_flow_log_df(power_log_df, "power_log.csv")
        update_street_statistics_csv(
            {f"{edge_to_street.get(e, 'Unknown')} ({e})": c for e, c in street_crossings.items()},
            "traffic_stats.csv"
        )
        traci.close()
        print(f"Simulation completed after {tick_counter} steps")


if __name__ == "__main__":
    main()