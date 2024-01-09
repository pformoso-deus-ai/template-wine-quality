import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    client = MlflowClient()

    runs = client.search_runs('2')
    max_run = max(runs, key=lambda run: run.data.metrics['accuracy'])
    max_accuracy_run_id = max_run.info.run_id
    
    
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        
        if mv["run_id"] == max_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            alias = f"{model_name}_{current_version}"        

            client.set_registered_model_alias(model_name, alias, current_version)   

            loaded_model = mlflow.pyfunc.load_model(logged_model)
            joblib.dump(loaded_model, model_dir)

        else:
            current_version = mv["version"]
            
            latest_mv = client.get_latest_versions(model_name, stages=["Staging"])[0]
            alias = f"{model_name}_{current_version}"
            client.set_registered_model_alias(model_name, alias, latest_mv.version)                          

def print_run_info(runs):
    for r in runs:
        print(f"run_id: {r.info.run_id}")
        print(f"lifecycle_stage: {r.info.lifecycle_stage}")
        print(f"metrics: {r.data.metrics}")                
        
        # Exclude mlflow system tags
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        print(f"tags: {tags}")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)