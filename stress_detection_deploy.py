
###Imports
import uuid
import pandas as pd
import mlflow

def generate_uuids(n):
    ids=[]
    for i in range(n):
        ids.append(str(uuid.uuid4()))       
    return ids

def load_model(model_name,stage): 
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
    return model
def read_dataFrame(input_file):
    df=pd.read_csv(input_file)
    df.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', \
             'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
    
    df['id']=generate_uuids(len(df))
    return df

def apply_model(input_file,output_file,model_name,stage):
    print(f'Reading data from {input_file}...')
    df=read_dataFrame(input_file)
    X_val = df.drop(['snoring_rate','limb_movement','eye_movement','stress_level','id'],axis=1)
    
    print(f'Loading the MLflow model which is in {stage}...')   
    model=load_model(model_name,stage)
    
    print(f'Predicting the results...')   
    y_pred=model.predict(X_val)
    
    print(f'Saving results to {output_file}...') 
    df_result=pd.DataFrame()
    df_result['id']=df['id']
    df_result['respiration_rate']=df['respiration_rate']
    df_result['body_temperature']=df['body_temperature']
    df_result['blood_oxygen']=df['blood_oxygen']
    df_result['heart_rate']=df['heart_rate']
    df_result['sleeping_hours']=df['sleeping_hours']
    df_result['predicted_stress_Level']=y_pred
    df_result.to_csv(output_file)


def run():
    input_file='./data/SaYoPillow.csv'
    output_file='./output/stress_output.csv'
    model_name="stress-level"
    stage="Production"
    apply_model(input_file,output_file,model_name,stage)
    print(f'Finished') 
    
if __name__ == '__main__':
    run()