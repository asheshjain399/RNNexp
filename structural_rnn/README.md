# Structural-RNN

## Prerequisites

Install [NeuralModels](https://github.com/asheshjain399/NeuralModels) and checkout its [srnn] branch

Download the motion capture data from [URL]. We originally obtained this data from H3.6m and processed it further to use. If you use this data set then please cite the original authors of H3.6m data set. Also check their lisencing conditions. 

Open ```basedir``` file and edit the path to where you have saved the data set. We will use the same directory to save the checkpoints of our trained model. Size of each checkpoint can go upto few 100's MB. So make sure that you have enough disk space before you start training your own model.

## Description

## Training Structural-RNN for human motion modeling on H3.6m data set

In order to train S-RNN on H3.6m you will run ```hyperParameterTuning.py```. In this script we have defined all the hyperparameters for training S-RNN, ERD [1], and LSTM-3LR [1]. 

```python hyperParameterTuning.py model```

```model``` can take one of the following values ```srnn``` or ```erd``` or ```lstm3lr```

```hyperParameterTuning.py``` will print name of a directory (```result_dir```) to your screen where the trained models and all results will be saved. 

```hyperParameterTuning.py``` will internally call ```trainDRA.py``` which will import ```NeuralModels``` and then build your model, and also train it. ```trainDRA.py``` will automatically dump results of motion forecasting (on test set) in ```result_dir```. 

Once training has finished, you will need to parse the forecasted motion using ```generateMotionData.py``` which is located in directory ```CRFProblems/H3.6m/```.

Inside ```generateMotionData.py``` you can specify the checkpoint directory that you want to parse. 

