import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os


if(__name__=='__main__'):

    losses_storage_dir = os.getcwd()
    losses_file_name   = sys.argv[1]

    #===================================================================================#

    losses_storage_file_path = losses_storage_dir + '/' + losses_file_name + '.json'
    with open(losses_storage_file_path, 'r') as json_file:
        data = json.load(json_file)
    num_training_samples_per_classification_array = np.array(data[0])
    final_epoch_loss_array                        = np.array(data[1])

    fig = plt.figure()
    plt.plot(2*num_training_samples_per_classification_array, final_epoch_loss_array)
    plt.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
    plt.xlabel('number of training samples')
    plt.ylabel('binary crossentropy')
    plt.ylim([0.02, 0.06])
    plt.title('Size of training data set size vs. training data set loss')

    plt.savefig('loss_vs_dataset_size.png')
