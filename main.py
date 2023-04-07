import argparse
import os
import re
import json
import tempfile
import yaml
import pandas as pd
from datetime import datetime
from google.cloud import aiplatform
import google.auth
import google.cloud.storage as gcs
import google.cloud.bigquery as bq
from kfp.v2 import compiler

from training_pipeline.pipeline import training_pipeline


_, PROJECT_ID = google.auth.default()
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
REPO_BRANCH = os.getenv("REPO_BRANCH")
BUCKET = "gs://" + re.sub(r"^gs://", "", os.getenv("BUCKET_ARTEFACTS"))
TENSORBOARD_ID = os.getenv("TENSORBOARD_ID")


def deploy_training_pipeline(request):

    payload = request.get_data(as_text=True)
    args = json.loads(payload)
    
    # example for args object
    # args = {
    #     "language": "en",
    # }
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    pipeline_cfg = cfg["pipeline_cfg"]
    model_cfg = cfg["model_cfg"][args["language"]]
    
    cfg = argparse.Namespace(**{
        **pipeline_cfg,
        **model_cfg,
        **args,
        "project_id": PROJECT_ID,
        "service_account": SERVICE_ACCOUNT,
        "repo_branch": REPO_BRANCH,
        "bucket": BUCKET,
        "tensorboard_id": TENSORBOARD_ID
    })
    
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = f"{cfg.bucket}/{cfg.gcs_training_pipeline_dir}/{cfg.language}/{cfg.language}_{dt}"
    model_name_output = cfg.model_name.split('/')[-1] + ".pt" # name of model output file
    endpoint_model_name = f"{cfg.endpoint_model_name}_{cfg.language}"
    
    setattr(cfg, "dt", dt)
    setattr(cfg, "working_dir", working_dir)
    setattr(cfg, "model_name_output", model_name_output)
    setattr(cfg, "endpoint_model_name", endpoint_model_name)
    

    pipeline_params = {
        "comp_params": cfg.__dict__,
        "model_info": {**args, **model_cfg, "working_dir": working_dir}
    }

    with tempfile.NamedTemporaryFile() as temp_f:
        pipeline_filename = f"{temp_f.name}.json"
        compiler.Compiler().compile(
            pipeline_func=training_pipeline,
            package_path=pipeline_filename
        )
        
    response = aiplatform.PipelineJob(
        display_name=working_dir,
        template_path=pipeline_filename,
        enable_caching=True,
        pipeline_root=working_dir,
        parameter_values=pipeline_params,
        project=cfg.project_id,
        location=cfg.pipeline_job_location
    )

    response.submit()

    print("Successfully deployed training pipeline")
    
    return "DONE"