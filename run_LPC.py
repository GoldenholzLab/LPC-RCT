import numpy as np
import keras.models as models
import json


def run_LPC(model1or2, percentChangePlacebo, percentChangeDrug):
    # calculates the LPC binary output
    # INPUTS:
    #  model1or2 = 1 or 2
    #  percentageChangePlacebo = a numpy 1D vector of percentage changes from all placebo patients
    # percentageChangeDrug  = a numpy 1D vector of percentage change from all drug patients
    # OUTPUTS:
    #  a binary 1 = true, 0 false
    #  it is assumed that the following statement will be true: len(percentChangePlacebo) == len(percentChangeDrug)

    # Constant values
    num_bins = 80
    hist_range = [-3, 1]

    modelForNV1FileName = 'NV_Model_1.h5'
    modelForNV2FileName = 'NV_Model_2.h5'
    thresholdsForNV1file = 'NV_model_1_thresholds.json'
    thresholdsForNV2file = 'NV_model_1_thresholds.json'

    possible_thresholds = np.arange(40, 460 + 40, 20)

    if(model1or2 == 1):
        deepRCT_model = models.load_model(modelForNV1FileName)
        with open(thresholdsForNV1file , 'r') as json_file:
            threshold_array = np.array(json.load(json_file))
    elif(model1or2 == 2):
        deepRCT_model = models.load_model(modelForNV2FileName)
        with open(thresholdsForNV2file , 'r') as json_file:
            threshold_array = np.array(json.load(json_file))

    [percentChangePlaceboHist, _]= \
        np.histogram(percentChangePlacebo, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)
    
    [percentChangeDrugHist, _] = \
        np.histogram(percentChangeDrug, 
                     bins=num_bins, 
                     range=hist_range, 
                     density=True)
    
    percentChangePlaceboHistKerasSample = np.zeros((1, num_bins, 1))
    percentChangeDrugHistKerasSample    = np.zeros((1, num_bins, 1))

    percentChangePlaceboHistKerasSample[0, :, 0] = percentChangePlaceboHist
    percentChangeDrugHistKerasSample[0, :, 0]    = percentChangeDrugHist

    prediction = np.squeeze(deepRCT_model.predict([percentChangePlaceboHistKerasSample, percentChangeDrugHistKerasSample]))

    threshold_index = np.argwhere(possible_thresholds == len(percentChangePlacebo))
    if(len(threshold_index) == 0):
        raise ValueError('LPC model does not have a threshold for a trial size of ' + str(len(percentChangePlacebo)))

    threshold = threshold_array[np.squeeze(threshold_index)]

    if(prediction > threshold):
        return 1
    else:
        return 0


if(__name__=='__main__'):

    model1or2 = 1
    percentChangePlacebo = np.array(100*[0.01])
    percentageChangeDrug = np.array(100*[1])

    print(run_LPC(model1or2, percentChangePlacebo, percentageChangeDrug))
