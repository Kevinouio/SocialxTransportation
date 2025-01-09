from transformers import pipeline
import traci
import xml.etree.ElementTree as ET


# Function to evaluate a rumor with LLM
def evaluate_rumor_with_llm(rumor, street_names):
    from transformers import pipeline

    # Initialize pipelines
    sentiment_pipe = pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True
    )
    classification_pipe = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )

    # Get sentiment analysis results
    sentiment_scores = sentiment_pipe(rumor)[0]  # Returns a list of dictionaries with scores for each label
    sentiment_results = {entry["label"]: entry["score"] for entry in sentiment_scores}

    # Determine overall sentiment
    if sentiment_results.get("negative", 0) > 0.40:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    # Street classification
    classification_result = classification_pipe(rumor, candidate_labels=street_names)
    relevant_streets = [
        street for street, score in zip(classification_result["labels"], classification_result["scores"]) if score > 0.5
    ]

    # Print results for debugging
    print(f"Relevant Streets: {relevant_streets}")
    print(f"Sentiment Results: {sentiment_results}")
    print(f"Overall Sentiment: {overall_sentiment}")

    return overall_sentiment, relevant_streets


# Function to extract street names and map them to edges and default danger levels
def get_edge_to_street_mapping(osm_file="osm.net.xml"):
    # Parse the XML file
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # Initialize dictionaries to store mapping
    edge_to_street = {}
    street_names = set()  # Use a set to avoid duplicate street names
    street_to_danger_level = {}

    # Iterate over all <edge> elements in the XML
    for edge in root.findall("edge"):
        edge_id = edge.get("id")  # Get the edge ID
        street_name = edge.get("name")  # Get the name attribute

        if street_name:  # Only consider edges with a valid name
            edge_to_street[edge_id] = street_name
            street_names.add(street_name)

    # Initialize danger levels for each street
    for street_name in street_names:
        street_to_danger_level[street_name] = 0  # Default danger level is 0

    # Convert street_names back to a sorted list
    return edge_to_street, sorted(street_names), street_to_danger_level


# Function to simulate a sample SUMO run
def simulate_sumo_run(rumor, net_file="osm.net.xml"):
    # Get edge-to-street mapping and default danger levels
    edge_to_street, street_names, street_to_danger_level = get_edge_to_street_mapping()

    # Evaluate the rumor
    sentiment_results, relevant_streets = evaluate_rumor_with_llm(rumor, street_names)

    # Update danger levels for relevant streets
    for street in relevant_streets:
        street_to_danger_level[street] = 1  # Mark relevant streets as dangerous

    # Print results
    print(f"Relevant Streets: {relevant_streets}")
    print(f"Sentiment Results: {sentiment_results}")
    print(f"Street Danger Levels: {street_to_danger_level}")

    # Restart SUMO to modify the simulation
    traci.start(["sumo", "-n", net_file])
    try:
        for edge, street_name in edge_to_street.items():
            danger_level = street_to_danger_level.get(street_name, 0)
            if danger_level > 0:
                print(f"Marking {edge} (part of {street_name}) as dangerous in the simulation.")
                # Example: Adjust edge parameters (e.g., speed or access)
                traci.edge.setMaxSpeed(edge, 0)  # Set speed to 0 for dangerous sections
    finally:
        traci.close()


# Main function for testing
if __name__ == '__main__':
    # Define a sample rumor
    sample_rumor = "There is traffic on Holden Boulevard"

    # Simulate a SUMO run
    simulate_sumo_run(sample_rumor)
