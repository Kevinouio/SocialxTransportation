import traci


def set_traffic_light_durations(tls_id, red_duration=60, green_duration=5):
    """
    Modifies the specified traffic light to have a red phase lasting for `red_duration`
    seconds and a green phase for `green_duration` seconds.

    Args:
        tls_id (str): The traffic light ID.
        red_duration (int): Duration for red phase (in seconds).
        green_duration (int): Duration for green phase (in seconds).
    """
    # Get current traffic light logic using the updated method
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logics:
        print(f"No traffic light logic found for {tls_id}")
        return
    logic = logics[0]

    # Here we define two phases: one red and one green.
    # The phase "rrrr" is all red and "GGGG" is all green.
    # Modify the states as needed for your network (for instance, if you have multiple lanes or conflicting movements).
    phase_red = traci.trafficlight.Phase(duration=red_duration, state="rrrr")
    phase_green = traci.trafficlight.Phase(duration=green_duration, state="GGGG")

    # Replace the existing phases with our custom ones.
    logic.phases = [phase_red, phase_green]

    # Set the new logic back to the traffic light.
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)
    print(f"Set {tls_id}: Red for {red_duration}s, Green for {green_duration}s")


def run_simulation_with_custom_tls(net_file, route_file, tls_id):
    """
    Runs the SUMO simulation and applies the custom traffic light durations.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the route file.
        tls_id (str): Traffic light ID to modify.
    """
    traci.start(["sumo-gui", "-n", net_file, "-r", route_file])
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Every step (or on a timed interval) update the traffic light
            set_traffic_light_durations(tls_id, red_duration=60, green_duration=5)
    finally:
        traci.close()
        print("Simulation ended.")


if __name__ == "__main__":
    # Make sure to use the correct network/route files and traffic light ID.
    # You might need to first run a script to list available traffic light IDs.
    run_simulation_with_custom_tls("osm_with_tls.net.xml", "osm.rou.xml", tls_id="my_tls")
