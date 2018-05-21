"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from network import Network
from pyswarm import pso

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """

    for network in networks:
        network.train(dataset)


def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def banana(vector):
    global nn_param_choices
    r = create_net_from_vec(nn_param_choices, vector)
    r.train('water')
    accuracy = r.accuracy
    print("Acc",accuracy)
    return 1 / accuracy


def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))
        
        

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def create_vector(nn_param_choices, network):
    """Generate vector params from the network"""
    vec = []
    n_param = network.network
    for key in n_param:
        temp = nn_param_choices[key].index(n_param[key])
        vec.append(temp)
    return vec
    
def create_net_from_vec(nn_param_choices, vector):
    network = {}
    vector = [int(x.round()) for x in vector]
    for i, key in enumerate(nn_param_choices):
        network[key] = nn_param_choices[key][vector[i]]
        
    r = Network(nn_param_choices)
    r.create_set(network)
    return r

nn_param_choices = {
        'nb_neurons': [5, 10 ],
        'nb_layers': [1, 2, 3],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam',  'adagrad',
                      'adadelta', 'adamax', ],
    }

def main():
    """Evolve a network."""
    generations = 10 # Number of times to evole the population.
    population = 10  # Number of networks in each generation.
    dataset = 'water'

    nn_param_choices = {
        'nb_neurons': [5, 10],
        'nb_layers': [1, 2, 3],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam',  'adagrad',
                      'adadelta', 'adamax', ],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    lower_bounds = [0] * 4
    upper_bounds = [len(nn_param_choices[x])-1 for x in nn_param_choices]
    
    xopt, fopt = pso(banana, lower_bounds, upper_bounds)
    
#    generate(generations, population, nn_param_choices, dataset)
    
#    optimizer = Optimizer(nn_param_choices)
#    networks = optimizer.create_population(population)
#    v = create_vector(nn_param_choices, networks[0])
#    n = create_net_from_vec(nn_param_choices, v)

if __name__ == '__main__':
    main()
