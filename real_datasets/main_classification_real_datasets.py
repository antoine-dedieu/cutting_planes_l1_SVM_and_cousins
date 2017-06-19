import sys
from compare_methods_classification_real_datasets import *


type_dataset = int(sys.argv[1])

n_average = 3
average_simulations_compare_methods_classification_real_datasets('logreg', type_dataset, n_average)
