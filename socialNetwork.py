import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution

#
def createWeights():
    g = nx.complete_graph(50)

    # Algorithmic Bias model
    model = op.AlgorithmicBiasModel(g)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", 0.32)
    config.add_model_parameter("gamma", 0)
    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(20) # Reduced iterations for brevity

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

    return CarWeights
def generateVisualGraphOfWeights(model,iterations):

    viz = OpinionEvolution(model, iterations)
    viz.plot("opinion_ev.pdf")