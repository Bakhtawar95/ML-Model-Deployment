This repository provides an example of dataset preprocessing,model training, model deployment in the form of a REST API as well as in batch mode.

## Project Structure

[data/SaYoPillow.csv](/data/SaYoPillow.csv)
        This dataset contains parameters like snoring rate, respiration rate, body temperature, limb movement rate, blood oxygen levels, eye movement, number of hours of sleep, heart rate and the coresponding Stress Levels (0- low/normal, 1 – medium low, 2- medium,3-medium high, 4 -high).

[stress_detection_train.py](/stress_detection_train.py)
                        This script reads the input data, does a minimal preprocessing and trains the models. For training, four classifiers are used, namely: Random Forest, Descision Tree,Gradient Boost and Naive Bayes. All the four trained models alongwith their parameters (RMSE etc.) are logged into MLflow. Then MLflow client is used to select the best model among these on the 
basis of RMSE (RMSE<0.05). In this way, one model is chosen. This model is then registered to model Registry of MLflow. Each time when we train the model, a new version of it is created. At the end MLflow client chooses the latest version of model from registry and promotes its stageto Production. This model will now be deployed.

[stress_detection_deploy.py](/stress_detection_deploy.py)
                          This script deploys the model in batch mode. For this, it first loads the model which was promoted to production from MLflow model registry. This model then predicts the output stress level values. This output is saved to stress_output.csv file.
                          
[API.py](/API.py)
       This contains Flask REST API of the model. This API receives the input details via the GUI, computes the predicted value using the saved model and prompts the output. 
       
[templates/index.html](/templates/index.html)
                    This HTML template serves as a GUI to enter the input details and displays the corresponding predicted stress level.
                    








