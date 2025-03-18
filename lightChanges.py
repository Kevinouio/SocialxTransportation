import traci

def list_traffic_lights():
    tls_ids = traci.trafficlight.getIDList()
    print("Traffic lights found:", tls_ids)
    return tls_ids

def set_traffic_light_to_red(tls_id, red_duration=60):
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logics:
        print(f"No traffic light logic found for {tls_id}")
        return
    logic = logics[0]
    red_phase = traci.trafficlight.Phase(duration=red_duration, state="rrrr")
    logic.phases = [red_phase]
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)
    print(f"Set {tls_id} to red for {red_duration}s")

def run_simulation_all_red(net_file, route_file):
    traci.start(["sumo-gui", "-n", net_file, "-r", route_file])
    try:
        tls_ids = list_traffic_lights()
        if not tls_ids:
            print("No traffic lights to modify. Exiting simulation.")
            return

        # Apply the red-only phase once at startup.
        for tls_id in tls_ids:
            set_traffic_light_to_red(tls_id, red_duration=1800)

        # Continue simulation without further updates.
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
    finally:
        traci.close()
        print("Simulation ended.")

if __name__ == "__main__":
    run_simulation_all_red("osm.net.xml", "osm.rou.xml")
