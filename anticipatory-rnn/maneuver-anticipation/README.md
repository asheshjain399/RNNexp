#How to train your model?

You maneuver-rnn.py to train you model. 

```
python maneuver-rnn.py index fold
```

```index``` is the id of the dataset and ```fold``` is the fold number. Before running this script make sure that ```path_to_dataset``` and ```path_to_checkpoints``` are correctly defined. You can additionally choose the kind of architecture to train by setting the parameter ```model_type```. For the time being some of these parameters are hardcoded in the script.

#How to test your model?

##Testing a single checkpoint
In order to evaluate at a single checkpoint see the script ```evaluateCheckpoints.py```. Usually I don't run this script directly. Instead I call it from ```runbatchprediction.py``` or ```generateBestResults.py```

##Generating predictions for all checkpoints and thresholds
Use ```runbatchprediction.py```. This script helps choosing a correct checkpoint and threshold value for each fold.

##Generating best numbers for the table
Use ```generateBestResults.py```. 

```
python generateBestResults.py maneuver_type model_id
```

This script generates best results for a given model. It reads the configuration file in chekpoints/```maneuver_type```/```model_id```
