# This script generates a diagram of the classifier model architecture
# credits: https://github.com/teha2/chemical_toxicology/blob/c22571d2f9b11060fd2d3f2dab5ff6a4ff1d350e/MIE%20NN%20Models%20Oct%202019/Model_Generator.py
from graphviz import Digraph


def draw_classifier_architecture():
    # Create a new directed graph
    graph = Digraph("ClassifierModel", node_attr={"shape": "record", "height": ".1"})

    # Input Layer
    graph.node("Input", "Input Layer\n(128 features)")

    # Hidden Layer 1
    graph.node("FC1", "Fully Connected Layer\n(128 neurons)")
    graph.node("BN1", "Batch Norm Layer\n(128)")
    graph.node("ReLU1", "Leaky ReLU Activation")
    graph.node("Dropout1", "Dropout Layer")

    # Hidden Layer 2
    graph.node("FC2", "Fully Connected Layer\n(128 neurons)")
    graph.node("BN2", "Batch Norm Layer\n(128)")
    graph.node("ReLU2", "Leaky ReLU Activation")
    graph.node("Dropout2", "Dropout Layer")

    # Hidden Layer 3
    graph.node("FC3", "Fully Connected Layer\n(128 neurons)")
    graph.node("BN3", "Batch Norm Layer\n(128)")
    graph.node("ReLU3", "Leaky ReLU Activation")
    graph.node("Dropout3", "Dropout Layer")

    # Output Layer
    graph.node("Output", "Output Layer\n(2 classes: STOP/GO)")

    # Connecting Input to First Hidden Layer
    graph.edge("Input", "FC1")
    graph.edge("FC1", "BN1")
    graph.edge("BN1", "ReLU1")
    graph.edge("ReLU1", "Dropout1")

    # Connecting First to Second Hidden Layer
    graph.edge("Dropout1", "FC2")
    graph.edge("FC2", "BN2")
    graph.edge("BN2", "ReLU2")
    graph.edge("ReLU2", "Dropout2")

    # Connecting Second to Third Hidden Layer
    graph.edge("Dropout2", "FC3")
    graph.edge("FC3", "BN3")
    graph.edge("BN3", "ReLU3")
    graph.edge("ReLU3", "Dropout3")

    # Connecting Third Hidden Layer to Output Layer
    graph.edge("Dropout3", "Output")

    # Render and save the graph
    graph.render("classifier_model_architecture", format="png")


# Call the function to generate the diagram
draw_classifier_architecture()
