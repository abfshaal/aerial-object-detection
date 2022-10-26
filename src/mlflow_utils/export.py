import pandas as pd 
from yaml import safe_load

import mlflow
import fire

def read_model_result(path:str):
  df = pd.read_csv(path)
  return df

def log_params(yaml_path):
  with open(yaml_path) as f:
    dict_params = safe_load(f)
  dict_params = dict_params['vars']
  params = dict()

  for d in dict_params:
    params.update(d)
  for k,v in params.items():
    if 'path' not in k:
      mlflow.log_param(k,v)

def log_metrics(df):
  df.columns = [val.lstrip() for val in df.columns]
  for index,row in df.iterrows():
    mlflow.log_metrics(**row.to_dict(),step=index)

def log_model(model_path:str):
  pass

def track_mlflow(result_path:str, model_path:str, yaml_path:str):
  mlflow.set_experiment('aerial_detection')
  with mlflow.start_run():
    df = read_model_result(result_path)
    log_params(yaml_path)
    log_metrics(df)
    log_model(model_path)


if __name__ == '__main__':
  fire.Fire(track_mlflow)
