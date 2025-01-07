import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
import xml.etree.ElementTree as ET
#
def createWeights(carTotal,time):
    g = nx.complete_graph(carTotal)

    # Algorithmic Bias model
    deez = op.AlgorithmicBiasModel(g)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", 0.32)
    config.add_model_parameter("gamma", 0)
    deez.set_initial_status(config)

    # Simulation execution
    iterations = deez.iteration_bunch(time) # Reduced iterations for brevity

    # Store opinions in a list of dictionaries
    opinion_history = []

    # List of weights for each car respectively
    CarWeights = []

    print(type(iterations[0]))

    for weights in iterations:
        opinion_history.append(weights["status"])

    print(opinion_history)
    for weights in opinion_history:
        iterWeights = []
        for i in range(50):
            iterWeights.append(weights[i])
        CarWeights.append(iterWeights)



    return deez, iterations, CarWeights


def generateVisualGraphOfWeights(model,iterations):

    viz = OpinionEvolution(model, iterations)
    viz.plot("opinion_ev.pdf")

if __name__ == "__main__":
    model, iterations, CarWeights = createWeights()
    generateVisualGraphOfWeights(model,iterations)