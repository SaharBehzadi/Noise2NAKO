import sys
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment for error with MacOS BigSur
# import the sddr module
from sddr import Sddr
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import torch
import os

def normalize(x):
    x = x - x.mean()
    x = x/x.std()
    return x

def integration_test_simple_gam():
    '''
    Integration test using a Simple GAM Poisson Distribution.
    The partial effects are estimated and compared with the ground truth 
    (only functional form: the terms are normalized before comparison)
    If the error is higher than a resonable value an error is raised.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = '../data/simple_gam/X.csv'
    target_path = '../data/simple_gam/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)

    output_dir = './outputs'

    #define Sddr parameters
    distribution  = 'Poisson'

    formulas = {'rate': '~1 + spline(x1, bs="bs",df=9) + spline(x2, bs="bs",df=9) + d1(x1) + d2(x2)'}
    deep_models_dict = {
    'd1': {
        'model': nn.Sequential(nn.Linear(1,15)),
        'output_shape': 15},
    'd2': {
        'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
        'output_shape': 8}
    }

    train_parameters = {
        'batch_size': 1000,
        'epochs': 1000,
        'degrees_of_freedom': {'rate': 6},
        'optimizer' : optim.RMSprop,
        'val_split': 0.15,
        'early_stop_epochs': 100,
        'early_stop_epsilon': 0.001
    }
    
    #initialize Sddr
    sddr = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)

    # train Sddr
    sddr.train(target=target, structured_data=data)
    
    #compute partial effects
    partial_effects_rate = sddr.eval('rate',plot=False)

    #normalize partial effects and compare with ground truth
    x = partial_effects_rate[0][0]
    y = normalize(partial_effects_rate[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated in simple GAM."
    
    x = partial_effects_rate[1][0]
    y = normalize(partial_effects_rate[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.02, "Partial effect not properly estimated in simple GAM."
    
    #compute partial effects on unseen data
    _, partial_effects_pred_rate = sddr.predict(data/2,clipping=True)
    
    #normalize partial effects and compare with ground truth
    x = partial_effects_pred_rate['rate'][0][0]
    y = normalize(partial_effects_pred_rate['rate'][0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated on unseen data in simple GAM."
    
    x = partial_effects_pred_rate['rate'][1][0]
    y = normalize(partial_effects_pred_rate['rate'][1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.02, "Partial effect not properly estimated on unseen data in simple GAM."


    
def integration_test_gamlss():
    '''
    Integration test using a GAMLSS - Logistic Distribution.
    The partial effects are estimated and compared with the ground truth 
    (only functional form: the terms are normalized before comparison)
    If the error is higher than a resonable value an error is raised.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = '../data/gamlss/X.csv'
    target_path = '../data/gamlss/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)

    output_dir = './outputs'

    #define Sddr parameters
    distribution  = 'Logistic'

    formulas = {'loc': '~1+spline(x1, bs="bs", df=4)+spline(x2, bs="bs",df=4) + d1(x1)+d2(x2)',
                'scale': '~1 + spline(x3, bs="bs",df=4) + spline(x4, bs="bs",df=4)'
                }

    deep_models_dict = {
    'd1': {
        'model': nn.Sequential(nn.Linear(1,15)),
        'output_shape': 15},
    'd2': {
        'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
        'output_shape': 8}
    }

    train_parameters = {
        'batch_size': 1000,
        'epochs': 200,
        'degrees_of_freedom': {'loc':4, 'scale':4},
        'optimizer' : optim.RMSprop,
        'val_split': 0.01
    }

    #initialize Sddr
    sddr = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    
    # train Sddr
    sddr.train(target=target, structured_data=data)
    
    #compute partial effects
    partial_effects_loc = sddr.eval('loc',plot=False)
    partial_effects_scale = sddr.eval('scale',plot=False)

    #normalize partial effects and compare with ground truth
    x = partial_effects_loc[0][0]
    y = normalize(partial_effects_loc[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()

    assert RMSE<0.12, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_loc[1][0]
    y = normalize(partial_effects_loc[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.1, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_scale[0][0]
    y = normalize(partial_effects_scale[0][1])

    y_target = normalize(x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.15, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_scale[1][0]
    y = normalize(partial_effects_scale[1][1])

    y_target = normalize(np.sin(4*x)) # ground truth: sinusoidal effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.4, "Partial effect not properly estimated in GAMLSS."
        
    #compute partial effects on unseen data
    _, partial_effects = sddr.predict(data/2,clipping=True, plot=False) 
    partial_effects_pred_loc = partial_effects['loc'] 
    partial_effects_pred_scale = partial_effects['scale']

    #normalize partial effects and compare with ground truth
    x = partial_effects_pred_loc[0][0]
    y = normalize(partial_effects_pred_loc[0][1])

    y_target = normalize(x**2) # ground truth: quadratic effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.25, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_pred_loc[1][0]
    y = normalize(partial_effects_pred_loc[1][1])

    y_target = normalize(-x) # ground truth: linear effect

    RMSE = (y-y_target).std()

    assert RMSE<0.1, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_pred_scale[0][0]
    y = normalize(partial_effects_pred_scale[0][1])

    y_target = normalize(x) # ground truth: linear effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.15, "Partial effect not properly estimated in GAMLSS."
    
    x = partial_effects_pred_scale[1][0]
    y = normalize(partial_effects_pred_scale[1][1])

    y_target = normalize(np.sin(4*x)) # ground truth: sinusoidal effect

    RMSE = (y-y_target).std()
    
    assert RMSE<0.4, "Partial effect not properly estimated in GAMLSS."


def integration_test_load_and_predict():
    '''
    Integration test of training and saving a GAMLSS model, then loading the same  model.
    The model is used to predict on unseen data right after training and once again after it
    is loaded. The results of the two must match for the test to pass.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = '../data/gamlss/X.csv'
    target_path = '../data/gamlss/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)
    train_data = data.iloc[:800]
    train_target = target.iloc[:800] #data.iloc[:800]
    test_data = data.iloc[800:]

    output_dir = './outputs'

    #define Sddr parameters
    distribution  = 'Logistic'

    formulas = {'loc': '~1+spline(x1, bs="bs", df=4)+spline(x2, bs="bs",df=4) + d1(x1)+d2(x2)',
                'scale': '~1 + spline(x3, bs="bs",df=4) + spline(x4, bs="bs",df=4)'
                }

    deep_models_dict = {
    'd1': {
        'model': nn.Sequential(nn.Linear(1,15)),
        'output_shape': 15},
    'd2': {
        'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
        'output_shape': 8}
    }

    train_parameters = {
        'batch_size': 1000,
        'epochs': 200,
        'degrees_of_freedom': {'loc':4, 'scale':4},
        'optimizer' : optim.RMSprop
    }
    #initialize Sddr
    sddr = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    
    # train Sddr
    sddr.train(target=train_target, structured_data=train_data)
    _, partial_effects = sddr.predict(test_data, clipping=True)
    sddr.save('temp_gamlss.pth')
    # load trained Sddr and predict
    pred_sddr = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    pred_sddr.load('./outputs/temp_gamlss.pth', train_data)
    _, partial_effects_loaded = pred_sddr.predict(test_data, clipping=True)
    # compare partial effects
    for param in partial_effects.keys():
        for partial_effect, partial_effect_loaded in zip(partial_effects[param], partial_effects_loaded[param]):
            abs_err = (partial_effect[1] - partial_effect_loaded[1])
            assert sum(abs_err) < 0.001, "Partial effect not same with original prediction and prediction after load for param %s and partial effect %s"%(param, partial_effect[0])
    os.remove('./outputs/temp_gamlss.pth')


def integration_test_load_and_resume():
    '''
    Integration test of training and saving a GAMLSS model, then loading the same  model.
    The model is used to continue training.
    '''
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #load data
    data_path = '../data/simple_gam/X.csv'
    target_path = '../data/simple_gam/Y.csv'

    data = pd.read_csv(data_path,delimiter=';')
    target = pd.read_csv(target_path)

    output_dir = './outputs'

    #define Sddr parameters
    distribution  = 'Poisson'

    formulas = {'rate': '~1 + spline(x1, bs="bs",df=9) + spline(x2, bs="bs",df=9) + d1(x1) + d2(x2)'}
    deep_models_dict = {
    'd1': {
        'model': nn.Sequential(nn.Linear(1,15)),
        'output_shape': 15},
    'd2': {
        'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
        'output_shape': 8}
    }

    train_parameters = {
        'batch_size': 1000,
        'epochs': 100,
        'degrees_of_freedom': {'rate': 6},
        'optimizer' : optim.RMSprop,
        'val_split': 0
    }
    
    #initialize Sddr
    sddr_100 = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)

    # train Sddr
    sddr_100.train(target=target, structured_data=data)
    #_, partial_effects = sddr.predict(test_data, clipping=True)
    sddr_100.save('temp_simple_gam.pth')
    # load trained Sddr and predict
    train_parameters['epochs'] = 500
    sddr_resume = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters)
    sddr_resume.load('./outputs/temp_simple_gam.pth', data)
    sddr_resume.train(target=target, structured_data=data, resume=True)
    loss_resume = sddr_resume.epoch_train_loss
    # train continuously
    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    deep_models_dict = {
    'd1': {
        'model': nn.Sequential(nn.Linear(1,15)),
        'output_shape': 15},
    'd2': {
        'model': nn.Sequential(nn.Linear(1,3),nn.ReLU(), nn.Linear(3,8)),
        'output_shape': 8}
    }

    train_parameters = {
        'batch_size': 1000,
        'epochs': 500,
        'degrees_of_freedom': {'rate': 6},
        'optimizer' : optim.RMSprop,
        'val_split': 0
    }
    
    sddr_500 = Sddr(output_dir=output_dir,
            distribution=distribution,
            formulas=formulas,
            deep_models_dict=deep_models_dict,
            train_parameters=train_parameters)
    sddr_500.train(target=target, structured_data=data)
    loss_500 = sddr_500.epoch_train_loss
    loss_dif = abs(loss_500 - loss_resume)
    assert loss_dif < 0.001, "Loss function not equal in two training methods"

    os.remove('./outputs/temp_simple_gam.pth')
        
    
def integration_test_mnist():
    '''
    Integration test with unstructed data.
    A mixed model is used that has structued and unstructured input.
    The unstructured input are mnist images. These mnist images are used in the formula and represent the number that is
    on the image. The test tests if the estimated numbers are on average (median) a monotonically increasing function of the
    true numbers on the mnist images
    '''


    #set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    #load data
    data_path = '../data/mnist_data/tab.csv'

    data = pd.read_csv(data_path,delimiter=',').loc[:1000,:]

    for i in data.index:

        data.loc[i,'groundtruth'] = np.sin(data.loc[i,'x1']) - 3*data.loc[i,'x2'] + data.loc[i,'x3']**4 + 3*data.loc[i,'y_true']


    data.loc[:,'groundtruth'] = data.loc[:,'groundtruth'] - data.loc[:,'groundtruth'].mean()

    output_dir = './outputs'

    unstructured_data = {
      'numbers' : {
        'path' : '../data/mnist_data/mnist_images',
        'datatype' : 'image'
      }
    }

    for i in data.index:
        data.loc[i,'numbers'] = f'img_{i}.jpg'#f'{data.id[i]}.jpg'


    #define Sddr parameters
    formulas = {'loc': '~ -1 + spline(x1, bs="bs", df=10) + x2 + dnn(numbers) + spline(x3, bs="bs", df=10)',
                'scale': '~1'
                }
    distribution  = 'Normal'

    deep_models_dict = {
    'dnn': {
        'model': nn.Sequential(nn.Flatten(1, -1),
                               nn.Linear(28*28,128),
                               nn.ReLU()),
        'output_shape': 128},
    }

    train_parameters = {
        'batch_size': 100,
        'epochs': 100,
        'degrees_of_freedom': {'loc':9.6, 'scale':9.6},
        'optimizer' : optim.RMSprop
    }

    #initialize Sddr
    sddr = Sddr(output_dir=output_dir,
                distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters,
                )

    # train Sddr
    sddr.train(structured_data=data,
               target="groundtruth",
               unstructured_data = unstructured_data)

    data_pred = data.loc[:,:]
    distribution_layer, partial_effect = sddr.predict(data_pred,
                                                      clipping=True,
                                                      plot=False, 
                                                      unstructured_data = unstructured_data)

    assert distribution_layer.scale[0]>0.7, "Scale too large in mnist test"

    data_pred2 = data.copy()

    data_pred2.loc[:,'x1'] = 0
    data_pred2.loc[:,'x2'] = 0
    data_pred2.loc[:,'x3'] = 0
    data_pred2

    distribution_layer, partial_effect = sddr.predict(data_pred2,
                                                      clipping=True, 
                                                      plot=False, 
                                                      unstructured_data = unstructured_data)

    data_pred2['predicted_number'] = distribution_layer.loc[:,:].numpy().flatten()


    predicted_numbers  = data_pred2.groupby('y_true').median().predicted_number
    maximum_deviation_mnist = ((predicted_numbers.loc[1:].to_numpy() - predicted_numbers.loc[:8].to_numpy())/3).min()

    assert maximum_deviation_mnist>0, "Predicted numbers for the mnist not monotonically increasing"
    
    
if __name__ == '__main__':
    
    # run integration tests
    print("Test with simple GAM")
    integration_test_simple_gam()  
    print("---------------------------")
    print("Passed tests for simple GAM")
    
    print("Test with GAMLSS")
    integration_test_gamlss()   
    print("-----------------------")
    print("Passed tests for GAMLSS")
    
    print("Test with MNIST data")
    integration_test_mnist()   
    print("-----------------------")
    print("Passed tests for MNIST data")
    
    print("Test loading a GAMLSS model and predicting")
    integration_test_load_and_predict()
    print("-----------------------")
    print("Passed tests for loading and predicting")
    
    print("Test loading a GAMLSS model and resuming training")
    integration_test_load_and_resume()
    print("-----------------------")
    print("Passed tests for loading and resuming training")
