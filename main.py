import fire

from src.mlflow_utils.export import track_mlflow

if __name__=='__main__':
  fire.Fire(track_mlflow)