from transformers import pipeline
import random

def propagate_rumor(model, rumor):
    iterations = model.iteration_bunch(1)
    statuses = iterations[-1]["status"]
    return statuses

def evaluate_rumor_with_llm(rumor, street_names):
    sentiment_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True)
    classification_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment_scores = sentiment_pipe(rumor)[0]
    sentiment_results = {entry["label"]: entry["score"] for entry in sentiment_scores}
    overall_sentiment = "negative" if sentiment_results.get("negative", 0) > sentiment_results.get("positive", 0) else "neutral"
    classification_result = classification_pipe(rumor, candidate_labels=street_names)
    relevant_streets = [street for street, score in zip(classification_result["labels"], classification_result["scores"]) if score > 0.5]
    print(f"Relevant Streets: {relevant_streets}")
    print(f"Sentiment Results: {sentiment_results}")
    print(f"Overall Sentiment: {overall_sentiment}")
    return overall_sentiment, relevant_streets

def generate_prompts_based_on_cars(cartotal, street_names):
    num_prompts = max(1, cartotal // 3)
    prompts = []
    for _ in range(num_prompts):
        event_type = random.choice(["active shooter", "fire"])
        street = random.choice(street_names)
        prompts.append(f"There is a {event_type} at {street}.")
    return prompts
