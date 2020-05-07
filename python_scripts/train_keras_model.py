from keras import Input
from keras.layers import Conv1D, MaxPooling1D, Flatten, concatenate, Dense, Dropout, LeakyReLU
import keras.models as models
import numpy as np
import time
import json
import sys
import os


def build_single_perceptron(num_bins):

    placebo_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='placebo_arm_histogram')
    placebo_arm_flattened_tensor  = Flatten(name='placebo_arm_flattened_layer')(placebo_arm_hist_input_tensor)

    drug_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='drug_arm_histogram')
    drug_arm_flattened_tensor  = Flatten(name='drug_arm_flattened_layer')(drug_arm_hist_input_tensor)

    concatatenated_placebo_and_drug_arm_hist_inputs_tensor = concatenate([placebo_arm_flattened_tensor, drug_arm_flattened_tensor])

    output_node = Dense(1, activation='sigmoid', name='output_node')(concatatenated_placebo_and_drug_arm_hist_inputs_tensor)

    deepRCT_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], output_node)
    deepRCT_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return deepRCT_model


def build_really_simple_model(num_bins,
                              dropout_rate):

    placebo_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='placebo_arm_histogram')
    placebo_arm_flattened_tensor  = Flatten(name='placebo_arm_flattened_layer')(placebo_arm_hist_input_tensor)

    drug_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='drug_arm_histogram')
    drug_arm_flattened_tensor  = Flatten(name='drug_arm_flattened_layer')(drug_arm_hist_input_tensor)

    concatatenated_placebo_and_drug_arm_hist_inputs_tensor = concatenate([placebo_arm_flattened_tensor, drug_arm_flattened_tensor])

    hidden_layer = Dense(80, activation='relu', name='hidden_layer')(concatatenated_placebo_and_drug_arm_hist_inputs_tensor)

    dropout_layer = Dropout(dropout_rate, name='hidden_layer_dropout')(hidden_layer)

    output_node = Dense(1, activation='sigmoid', name='output_node')(dropout_layer)

    deepRCT_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], output_node)
    deepRCT_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return deepRCT_model


def build_simple_model(num_bins,
                       CNNet_1D_num_filters, 
                       CNNet_1D_kernel_size, 
                       Leaky_ReLU_neg_slope, 
                       dropout_rate):

    placebo_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='placebo_arm_histogram')
    placebo_arm_1D_conv_tensor    = Conv1D(CNNet_1D_num_filters, CNNet_1D_kernel_size, name='placebo_arm_1D_CNN')(placebo_arm_hist_input_tensor)
    placebo_arm_1D_conv_tensor    = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='placebo_arm_1D_CNN_activation')(placebo_arm_1D_conv_tensor)
    placebo_arm_1D_conv_tensor    = MaxPooling1D(CNNet_1D_kernel_size, name='placebo_arm_1D_CNN_max_pooling')(placebo_arm_1D_conv_tensor)
    placebo_arm_flattened_tensor  = Flatten(name='placebo_arm_flattened_layer')(placebo_arm_1D_conv_tensor)

    drug_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='drug_arm_histogram')
    drug_arm_1D_conv_tensor    = Conv1D(CNNet_1D_num_filters, CNNet_1D_kernel_size, name='drug_arm_1D_CNN')(drug_arm_hist_input_tensor)
    drug_arm_1D_conv_tensor    = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='drug_arm_1D_CNN_activation')(drug_arm_1D_conv_tensor)
    drug_arm_1D_conv_tensor    = MaxPooling1D(CNNet_1D_kernel_size, name='drug_arm_1D_CNN_max_pooling')(drug_arm_1D_conv_tensor)
    drug_arm_flattened_tensor  = Flatten(name='drug_arm_flattened_layer')(drug_arm_1D_conv_tensor)

    concatenated_placebo_and_drug_tensor = concatenate([placebo_arm_flattened_tensor, drug_arm_flattened_tensor])
    output_tensor = Dense(40, name='MLP_layer_1_nodes')(concatenated_placebo_and_drug_tensor)
    output_tensor = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='MLP_layer_1_activation')(output_tensor)
    output_tensor = Dropout(dropout_rate, name='MLP_layer_1_dropout')(output_tensor)
    output_tensor = Dense(1, activation='sigmoid', name='sigmoidal_output')(output_tensor)

    deepRCT_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], output_tensor)
    deepRCT_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return deepRCT_model


def build_intermediate_model(num_bins,
                             CNNet_1D_num_filters, 
                             CNNet_1D_kernel_size, 
                             Leaky_ReLU_neg_slope, 
                             dropout_rate):

    placebo_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='placebo_arm_histogram')
    placebo_arm_1D_conv_tensor    = Conv1D(CNNet_1D_num_filters, CNNet_1D_kernel_size, name='placebo_arm_1D_CNN')(placebo_arm_hist_input_tensor)
    placebo_arm_1D_conv_tensor    = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='placebo_arm_1D_CNN_activation')(placebo_arm_1D_conv_tensor)
    placebo_arm_1D_conv_tensor    = MaxPooling1D(CNNet_1D_kernel_size, name='placebo_arm_1D_CNN_max_pooling')(placebo_arm_1D_conv_tensor)
    placebo_arm_flattened_tensor  = Flatten(name='placebo_arm_flattened_layer')(placebo_arm_1D_conv_tensor)

    drug_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='drug_arm_histogram')
    drug_arm_1D_conv_tensor    = Conv1D(CNNet_1D_num_filters, CNNet_1D_kernel_size, name='drug_arm_1D_CNN')(drug_arm_hist_input_tensor)
    drug_arm_1D_conv_tensor    = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='drug_arm_1D_CNN_activation')(drug_arm_1D_conv_tensor)
    drug_arm_1D_conv_tensor    = MaxPooling1D(CNNet_1D_kernel_size, name='drug_arm_1D_CNN_max_pooling')(drug_arm_1D_conv_tensor)
    drug_arm_flattened_tensor  = Flatten(name='drug_arm_flattened_layer')(drug_arm_1D_conv_tensor)

    concatenated_placebo_and_drug_tensor = concatenate([placebo_arm_flattened_tensor, drug_arm_flattened_tensor])
    output_tensor = Dense(40, name='MLP_layer_1_nodes')(concatenated_placebo_and_drug_tensor)
    output_tensor = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='MLP_layer_1_activation')(output_tensor)
    output_tensor = Dropout(dropout_rate, name='MLP_layer_1_dropout')(output_tensor)
    output_tensor = Dense(20, name='MLP_layer_2_nodes')(output_tensor)
    output_tensor = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='MLP_layer_2_activation')(output_tensor)
    output_tensor = Dropout(dropout_rate, name='MLP_layer_2_dropout')(output_tensor)
    output_tensor = Dense(1, activation='sigmoid', name='sigmoidal_output')(output_tensor)

    deepRCT_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], output_tensor)
    deepRCT_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return deepRCT_model


def build_complicated_model(num_bins,
                            CNNet_1D_num_filters, 
                            CNNet_1D_kernel_size, 
                            Leaky_ReLU_neg_slope, 
                            dropout_rate):

    placebo_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='placebo_arm_histogram')
    placebo_arm_1D_conv_tensor    = Conv1D(CNNet_1D_num_filters, CNNet_1D_kernel_size, name='placebo_arm_1D_CNN')(placebo_arm_hist_input_tensor)
    placebo_arm_1D_conv_tensor    = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='placebo_arm_1D_CNN_activation')(placebo_arm_1D_conv_tensor)
    placebo_arm_1D_conv_tensor    = MaxPooling1D(CNNet_1D_kernel_size, name='placebo_arm_1D_CNN_max_pooling')(placebo_arm_1D_conv_tensor)
    placebo_arm_flattened_tensor  = Flatten(name='placebo_arm_flattened_layer')(placebo_arm_1D_conv_tensor)

    drug_arm_hist_input_tensor = Input(shape=(num_bins, 1), name='drug_arm_histogram')
    drug_arm_1D_conv_tensor    = Conv1D(CNNet_1D_num_filters, CNNet_1D_kernel_size, name='drug_arm_1D_CNN')(drug_arm_hist_input_tensor)
    drug_arm_1D_conv_tensor    = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='drug_arm_1D_CNN_activation')(drug_arm_1D_conv_tensor)
    drug_arm_1D_conv_tensor    = MaxPooling1D(CNNet_1D_kernel_size, name='drug_arm_1D_CNN_max_pooling')(drug_arm_1D_conv_tensor)
    drug_arm_flattened_tensor  = Flatten(name='drug_arm_flattened_layer')(drug_arm_1D_conv_tensor)

    concatenated_placebo_and_drug_tensor = concatenate([placebo_arm_flattened_tensor, drug_arm_flattened_tensor])
    output_tensor = Dense(40, name='MLP_layer_1_nodes')(concatenated_placebo_and_drug_tensor)
    output_tensor = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='MLP_layer_1_activation')(output_tensor)
    output_tensor = Dropout(dropout_rate, name='MLP_layer_1_dropout')(output_tensor)
    output_tensor = Dense(20, name='MLP_layer_2_nodes')(output_tensor)
    output_tensor = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='MLP_layer_2_activation')(output_tensor)
    output_tensor = Dropout(dropout_rate, name='MLP_layer_2_dropout')(output_tensor)
    output_tensor = Dense(10, name='MLP_layer_3_nodes')(output_tensor)
    output_tensor = LeakyReLU(alpha=Leaky_ReLU_neg_slope, name='MLP_layer_3_activation')(output_tensor)
    output_tensor = Dropout(dropout_rate, name='MLP_layer_3_dropout')(output_tensor)
    output_tensor = Dense(1, activation='sigmoid', name='sigmoidal_output')(output_tensor)

    deepRCT_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], output_tensor)
    deepRCT_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return deepRCT_model


def build_chosen_model(model_type,
                       num_bins,
                       dropout_rate=None,
                       CNNet_1D_num_filters=None,
                       CNNet_1D_kernel_size=None,
                       Leaky_ReLU_neg_slope=None):

    if(model_type == 'single_perceptron'):

        deepRCT_model = build_single_perceptron(num_bins)

        deepRCT_model_file_name = 'only_one_model'

    elif(model_type == '3_layer'):

        deepRCT_model = \
            build_really_simple_model(num_bins,
                                      dropout_rate)

        deepRCT_model_file_name = 'dropout_' + str(int(100*dropout_rate)) + '%'

    elif(model_type == 'simple'):

        deepRCT_model = \
            build_simple_model(num_bins,
                               CNNet_1D_num_filters, 
                               CNNet_1D_kernel_size, 
                               Leaky_ReLU_neg_slope, 
                               dropout_rate)

        deepRCT_model_file_name = 'dropout_'           + str(int(100*dropout_rate))         + \
                                  '%_num_filters_'     +     str(CNNet_1D_num_filters)      + \
                                  '_kernel_size_'      +     str(CNNet_1D_kernel_size)      + \
                                  '_leaky_relu_slope_' + str(int(10*Leaky_ReLU_neg_slope))

    elif(model_type == 'intermediate'):

        deepRCT_model = \
            build_intermediate_model(num_bins,
                                     CNNet_1D_num_filters, 
                                     CNNet_1D_kernel_size, 
                                     Leaky_ReLU_neg_slope, 
                                     dropout_rate)

        deepRCT_model_file_name = 'dropout_'           + str(int(100*dropout_rate))         + \
                                  '%_num_filters_'     +     str(CNNet_1D_num_filters)      + \
                                  '_kernel_size_'      +     str(CNNet_1D_kernel_size)      + \
                                  '_leaky_relu_slope_' + str(int(10*Leaky_ReLU_neg_slope))

    elif(model_type == 'complicated'):

        deepRCT_model = \
            build_complicated_model(num_bins,
                                    CNNet_1D_num_filters, 
                                    CNNet_1D_kernel_size, 
                                    Leaky_ReLU_neg_slope, 
                                    dropout_rate)
        
        deepRCT_model_file_name = 'dropout_'           + str(int(100*dropout_rate))         + \
                                  '%_num_filters_'     +     str(CNNet_1D_num_filters)      + \
                                  '_kernel_size_'      +     str(CNNet_1D_kernel_size)      + \
                                  '_leaky_relu_slope_' + str(int(10*Leaky_ReLU_neg_slope))

    else:
        
        raise ValueError(model_type + ' is not valid')
    
    return [deepRCT_model,
            deepRCT_model_file_name]


def retrieve_training_data(training_data_dir,
                           training_samples_file_name,
                           num_training_samples_per_classification):

    training_data_file_path = training_data_dir + '/' + training_samples_file_name + '.json'

    with open(training_data_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    placebo_arm_hists = np.array(data[0])[:2*num_training_samples_per_classification, :, :]
    drug_arm_hists    = np.array(data[1])[:2*num_training_samples_per_classification, :, :]
    labels            = np.array(data[2])[:2*num_training_samples_per_classification]

    return [placebo_arm_hists,
            drug_arm_hists,
            labels]


def store_trained_model(model_storage_dir,
                        model_storage_folder_name,
                        model_type,
                        deepRCT_model_file_name,
                        deepRCT_model,
                        start_time_in_seconds):

    model_storage_dir = model_storage_dir + '/' + model_storage_folder_name
    if( not os.path.isdir(model_storage_dir) ):
        os.mkdir(model_storage_dir)
    
    model_storage_dir = model_storage_dir + '/' + model_type
    if( not os.path.isdir(model_storage_dir) ):
        os.mkdir(model_storage_dir)
    
    model_storage_file_path = model_storage_dir + '/' + deepRCT_model_file_name + '.h5'
    training_time_file_path = model_storage_dir + '/' + deepRCT_model_file_name + '_runtimes.txt'

    deepRCT_model.save(model_storage_file_path)

    stop_time_in_seconds = time.time()
    total_runtime_in_minutes_str = str( np.round((stop_time_in_seconds - start_time_in_seconds)/60, 3) ) + ' minutes of training'

    with open(training_time_file_path, 'w+') as text_file:
        text_file.write('\n' + total_runtime_in_minutes_str)


def get_inputs():

    print(sys.argv)

    training_samples_file_name = sys.argv[1]
    model_storage_folder_name  = sys.argv[2]

    num_training_samples_per_classification = int(sys.argv[3])

    model_type = sys.argv[4]

    dropout_rate = None
    CNNet_1D_num_filters = None
    CNNet_1D_kernel_size = None
    Leaky_ReLU_neg_slope = None

    if(len(sys.argv) > 5):
        dropout_rate = float(sys.argv[5])

    if(len(sys.argv) > 6):
        CNNet_1D_num_filters =   int(sys.argv[6])
        CNNet_1D_kernel_size =   int(sys.argv[7])
        Leaky_ReLU_neg_slope = float(sys.argv[8])

    #========================================================================================#

    training_data_dir = os.getcwd()
    model_storage_dir = os.getcwd()

    num_bins = 80

    batch_size = 100
    num_epochs = 50

    return [training_samples_file_name,
            model_storage_folder_name,
            model_type,
            num_bins,
            dropout_rate,
            CNNet_1D_num_filters,
            CNNet_1D_kernel_size,
            Leaky_ReLU_neg_slope,
            training_data_dir,
            model_storage_dir,
            num_training_samples_per_classification,
            batch_size,
            num_epochs]


def main():

    start_time_in_seconds = time.time()

    [training_samples_file_name,
     model_storage_folder_name,
     model_type,
     num_bins,
     dropout_rate,
     CNNet_1D_num_filters,
     CNNet_1D_kernel_size,
     Leaky_ReLU_neg_slope,
     training_data_dir,
     model_storage_dir,
     num_training_samples_per_classification,
     batch_size,
     num_epochs] = \
         get_inputs()

    [deepRCT_model,
     deepRCT_model_file_name] = \
        build_chosen_model(model_type,
                           num_bins,
                           dropout_rate,
                           CNNet_1D_num_filters,
                           CNNet_1D_kernel_size,
                           Leaky_ReLU_neg_slope)

    [placebo_arm_hists,
     drug_arm_hists,
     labels] = \
         retrieve_training_data(training_data_dir,
                                training_samples_file_name,
                                num_training_samples_per_classification)
    
    deepRCT_model.fit([placebo_arm_hists, drug_arm_hists], labels, batch_size=batch_size, epochs=num_epochs)

    store_trained_model(model_storage_dir,
                        model_storage_folder_name,
                        model_type,
                        deepRCT_model_file_name,
                        deepRCT_model,
                        start_time_in_seconds)


if(__name__=='__main__'):
    
    main()
 