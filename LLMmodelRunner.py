from transformers import pipeline

def sentimentAnalysis(text):
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    # Run the model

    result = pipe(text)
    print(result)
    return result
def streetClassification(text, streetNames):
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
    prompt = ("Does this rumor relate to any street that is listed below or have a mention of a street at all?"
              + streetNames)



    result = pipe(prompt)
    print(result)
    return result


if __name__ == '__main__':
   text =  input("What rumor would you like to spread? \nEnter:")
   print(sentimentAnalysis(text))
   print(streetClassification(text))