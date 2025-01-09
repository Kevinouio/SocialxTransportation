import requests

# Server Configuration
SERVER_IP = "http://130.18.208.34:5000"
 # Replace 'your-server-ip' with the actual IP or domain.

# Initialize the social network
def initialize_social_network_on_server(car_total):
    try:
        response = requests.post(f"{SERVER_IP}/initialize", json={"car_total": car_total})
        if response.status_code == 200:
            print("Social network initialized:", response.json())
        else:
            print("Failed to initialize social network:", response.text)
    except requests.exceptions.RequestException as e:
        print("Error while trying to connect to the server:", e)

# Propagate the rumor
def propagate_rumor_on_server(rumor, steps=1):
    try:
        response = requests.post(f"{SERVER_IP}/propagate", json={"rumor": rumor, "steps": steps})
        if response.status_code == 200:
            print("Rumor propagated successfully.")
            return response.json()  # Returns the statuses
        else:
            print("Failed to propagate rumor:", response.text)
    except requests.exceptions.RequestException as e:
        print("Error while trying to connect to the server:", e)
    return {}

# Example Usage
if __name__ == "__main__":
    # Number of nodes (cars) in the social network
    car_total = 3000

    # Initialize the social network on the server
    initialize_social_network_on_server(car_total)

    # Propagate a rumor
    rumor = "Active shooter on Main Street"
    statuses = propagate_rumor_on_server(rumor, steps=1)

    # Print the returned statuses
    if statuses:
        print("Node statuses:", statuses)
    else:
        print("No statuses received.")
