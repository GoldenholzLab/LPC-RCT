from train_keras_model import build_single_perceptron
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_model_and_load_training_data(training_data_dir,
                                        training_samples_file_name):

    training_data_file_path = training_data_dir + '/' + training_samples_file_name + '.json'
    
    single_perceptron_model = build_single_perceptron(80)

    with open(training_data_file_path, 'r') as training_data_json_file:
        data = json.load(training_data_json_file)
    
    placebo_arm_hists = np.array(data[0])
    drug_arm_hists    = np.array(data[1])
    labels            = np.array(data[2])

    return [single_perceptron_model,
            placebo_arm_hists,
            drug_arm_hists,
            labels]


def training_set_size_vs_loss(single_perceptron_model,
                              placebo_arm_hists,
                              drug_arm_hists,
                              labels,
                              batch_size,
                              num_epochs,
                              num_training_samples_per_classification_step,
                              final_num_training_samples_per_classification):

    num_training_samples_per_classification_array = \
        np.arange(num_training_samples_per_classification_step,
                  final_num_training_samples_per_classification + num_training_samples_per_classification_step,
                  num_training_samples_per_classification_step)
    
    num_training_set_sizes = \
        int(final_num_training_samples_per_classification/num_training_samples_per_classification_step)
    
    final_epoch_loss_array = np.zeros(num_training_set_sizes)
    
    for num_training_samples_per_classification_index in range(num_training_set_sizes):

        num_training_samples_per_classification = \
            num_training_samples_per_classification_array[num_training_samples_per_classification_index]

        tmp_placebo_arm_hists = placebo_arm_hists[:2*num_training_samples_per_classification, :, :]
        tmp_drug_arm_hists    =    drug_arm_hists[:2*num_training_samples_per_classification, :, :]
        tmp_labels            =            labels[:2*num_training_samples_per_classification]

        history = \
            single_perceptron_model.fit([tmp_placebo_arm_hists, tmp_drug_arm_hists],
                                         tmp_labels, batch_size=batch_size, epochs=num_epochs)

        final_epoch_loss_array[num_training_samples_per_classification_index] = history.history['loss'][num_epochs - 1]
    
    return [num_training_samples_per_classification_array,
            final_epoch_loss_array]


def store_losses_over_training_set_sizes(num_training_samples_per_classification_array, 
                                         final_epoch_loss_array,
                                         losses_storage_dir,
                                         losses_file_name):

    losses_storage_file_path = losses_storage_dir + '/' + losses_file_name + '.json'

    with open(losses_storage_file_path, 'w+') as json_file:
        data = []
        data.append(num_training_samples_per_classification_array.tolist())
        data.append(final_epoch_loss_array.tolist())
        json.dump(data,json_file)


def get_inputs():

    training_data_dir  = os.getcwd()
    losses_storage_dir = os.getcwd()

    training_samples_file_name = '200000_weekly_level_15_training_samples'
    losses_file_name           = 'training_set_size_losses'

    num_training_samples_per_classification_step  =   5000
    final_num_training_samples_per_classification = 100000

    batch_size = 100
    num_epochs = 50

    return [training_data_dir,
            training_samples_file_name,
            losses_storage_dir,
            losses_file_name,
            num_training_samples_per_classification_step,
            final_num_training_samples_per_classification,
            batch_size,
            num_epochs]


def main():

    [training_data_dir,
     training_samples_file_name,
     losses_storage_dir,
     losses_file_name,
     num_training_samples_per_classification_step,
     final_num_training_samples_per_classification,
     batch_size,
     num_epochs] = \
         get_inputs()

    [single_perceptron_model,
     placebo_arm_hists,
     drug_arm_hists,
     labels] = \
         create_model_and_load_training_data(training_data_dir,
                                             training_samples_file_name)

    [num_training_samples_per_classification_array,
     final_epoch_loss_array] = \
         training_set_size_vs_loss(single_perceptron_model,
                                   placebo_arm_hists,
                                   drug_arm_hists,
                                   labels,
                                   batch_size,
                                   num_epochs,
                                   num_training_samples_per_classification_step,
                                   final_num_training_samples_per_classification)
    
    store_losses_over_training_set_sizes(num_training_samples_per_classification_array, 
                                         final_epoch_loss_array,
                                         losses_storage_dir,
                                         losses_file_name)


if(__name__=='__main__'):

    start_time_in_seconds = time.time()

    main()

    stop_time_in_seconds = time.time()
    total_runtime_in_seconds = stop_time_in_seconds - start_time_in_seconds
    total_runtime_in_minutes = total_runtime_in_seconds/60
    total_runtime_in_minutes_str = str(np.round(total_runtime_in_minutes, 3)) + ' minutes'

    print(total_runtime_in_minutes_str)

