from transformers import pipeline
import random

def evaluate_rumor_with_llm(rumor, street_names):
    """
    Evaluates a rumor prompt with LLMs for sentiment and street classification.
    Args:
        rumor (str): The rumor prompt to evaluate.
        street_names (list): A list of street names for classification.
    Returns:
        tuple: Overall sentiment and relevant streets.
    """
    # Initialize pipelines
    sentiment_pipe = pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True
    )
    classification_pipe = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )

    # Get sentiment analysis results
    sentiment_scores = sentiment_pipe(rumor)[0]
    sentiment_results = {entry["label"]: entry["score"] for entry in sentiment_scores}

    # Determine overall sentiment
    if sentiment_results.get("negative", 0) > 0.35:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    # Street classification
    classification_result = classification_pipe(rumor, candidate_labels=street_names)
    relevant_streets = [
        street for street, score in zip(classification_result["labels"], classification_result["scores"]) if score > 0.5
    ]

    print(f"Relevant Streets: {relevant_streets}")
    print(f"Sentiment Results: {sentiment_results}")
    print(f"Overall Sentiment: {overall_sentiment}")

    return overall_sentiment, relevant_streets


def generate_prompts_based_on_cars(cartotal, street_names):
    """
    Generates prompts based on the total number of cars.
    Args:
        cartotal (int): The total number of cars in the simulation.
        street_names (list): A list of street names.
    Returns:
        list: A list of generated prompts.
    """
    num_prompts = max(1, cartotal // 3)  # Ensure at least one prompt is generated
    prompts = []

    for _ in range(num_prompts):
        event_type = random.choice(["active shooter", "traffic jam"])
        street = random.choice(street_names)
        prompts.append(f"There is a {event_type} at {street}.")

    return prompts


def simulate_time_steps(cartotal, street_names, time_steps):
    """
    Simulates time steps, pulling and evaluating prompts at each step.
    Args:
        cartotal (int): Total number of cars in the simulation.
        street_names (list): List of street names.
        time_steps (int): Total number of time steps to simulate.
    """
    # Generate prompts
    prompts = generate_prompts_based_on_cars(cartotal, street_names)

    print("\nGenerated Prompts:")
    for prompt in prompts:
        print(prompt)

    print("\nSimulation Results:")
    for step in range(1, time_steps + 1):
        if not prompts:
            print(f"Time Step {step}: No more prompts to evaluate.")
            break

        # Pull a prompt for evaluation
        prompt = prompts.pop(0)
        sentiment, relevant_streets = evaluate_rumor_with_llm(prompt, street_names)

        # Display evaluation results
        print(f"Time Step {step}:")
        print(f"  Prompt: {prompt}")
        print(f"  Sentiment: {sentiment}")
        print(f"  Relevant Streets: {relevant_streets}")


if __name__ == '__main__':
    # Initialize variables
    cartotal = int(input("Enter the total number of cars: "))
    street_names = ["Main Street", "Elm Street", "Oak Avenue", "Pine Road"]
    time_steps = int(input("Enter the total number of time steps: "))

    # Run the simulation
    simulate_time_steps(cartotal, street_names, time_steps)
