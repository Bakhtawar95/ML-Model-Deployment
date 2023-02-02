

###Imports

import pandas as pd
import uuid
import pickle
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType



# # MLflow Experiment Setup


###Globals

MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("stress-level-detection")
model_name="stress-level-prediction"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# # Reading Input data




def generate_uuids(n):
    ids=[]
    for i in range(n):
        ids.append(str(uuid.uuid4()))       
    return ids

def read_dataFrame(input_file):
    df=pd.read_csv(input_file)
    df.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen',              'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
    
    df['id']=generate_uuids(len(df))
    X = df.drop(['stress_level','id'],axis=1)
    y = df.stress_level
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, 
                                                    random_state=0)
    return X_train,y_train,X_test,y_test


# # Model Training and logging to MLflow



def training_logging(X_train,y_train,X_test,y_test):
    classifiers = [('Random Forest', RandomForestClassifier()), 
                   ('Decision Tree Classifier', DecisionTreeClassifier()), 
                   ('Gradient Boost Classifier',GradientBoostingClassifier(n_estimators=20, random_state = 0)),
                   ('Naive Bayes', GaussianNB())]


    for clf_name, clf in classifiers:
        mlflow.sklearn.autolog()                
        with mlflow.start_run():                
            mlflow.set_tag("model",f"{clf_name}")
            model=clf.fit(X_train, y_train)
            with open(f'models/{clf_name}.bin','wb') as f_out:
                pickle.dump((model),f_out)
            y_pred = clf.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred,squared="False")  
            print("RMSE of ", clf_name, " is: %.4f" % rmse)
            mlflow.log_metric("rmse",rmse)
            mlflow.log_artifact(local_path=f"models/{clf_name}.bin",artifact_path="models_pickle")
            accuracy = accuracy_score(y_test, y_pred)
            print("Test Accuracy of ", clf_name, " is ", accuracy)


# # MLflow Client to select the best model



def select_model():
    best_run = client.search_runs(
        experiment_ids='12',
        filter_string="metrics.rmse<0.01",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )

    for run in best_run:
        best_run_id=run.info.run_id
        best_run_rmse=run.data.metrics['rmse']
        print(f"run id of best run: {best_run_id}, rmse: {best_run_rmse}")
        return best_run_id


# # Registering the best model




def register_model(best_run_id):
    model_uri=f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri,name=model_name)
    #print("Model regstered successfully")


# # Promoting latest version of best model to Production




def promote_model():
    global best_model_latest_version
    latest_ver=client.get_latest_versions(name=model_name)
    new_stage="Production"
    date=datetime.today().date()
    for version in latest_ver:
        best_model_latest_version=version.version
        current_stage=version.current_stage
    
    client.transition_model_version_stage(
        name=model_name,
        version=best_model_latest_version,
        stage=new_stage,
        archive_existing_versions=False
    )
    client.update_model_version(
        name=model_name,
        version=best_model_latest_version,
        description=f"The model version {best_model_latest_version} was transitioned to {new_stage} on {date}"
    )
    print(f"The latest version of model is {best_model_latest_version}.It is promoted successfully to {new_stage}")





def train():
    input_file='./data/SaYoPillow.csv'
    X_train,y_train,X_test,y_test=read_dataFrame(input_file)
    training_logging(X_train,y_train,X_test,y_test)
    time.sleep(3)
    best_run_id=select_model()
    time.sleep(3)
    register_model(best_run_id)
    time.sleep(3)
    promote_model()
    print("Finished")
    



if __name__ == '__main__':
    train()

