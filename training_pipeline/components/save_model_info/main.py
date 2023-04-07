from typing import NamedTuple


def save_model_info(
    comp_params: dict,
    model_info: dict,
    model_resource_name: str
) -> NamedTuple('save_model_info_outputs', [
    ('DUMMY', str)
]):

    import argparse
    import json
    from google.cloud import storage as gcs
    
    cfg = argparse.Namespace(**comp_params)

    model_info["model_resource_name"] = model_resource_name

    client = gcs.Client(project=cfg.project_id)
    bucket_name = cfg.bucket.split("gs://")[-1]
    bucket = client.bucket(bucket_name)
    
    target_dir = f"{cfg.model_info_dir}/{cfg.language}/{cfg.language}_{{}}" # contains information for latest model
    wd = f"{cfg.working_dir[len(cfg.bucket)+1:]}/{{}}" # working dir of current training run
    
    # copy mapping objects created during training to model info dir
    # https://cloud.google.com/storage/docs/copying-renaming-moving-objects#storage-copy-object-python
    blob_model_info = bucket.blob(wd.format(cfg.output_file_model_info))
    blob_model_info.upload_from_string(json.dumps(model_info))
    bucket.copy_blob(blob_model_info, bucket, target_dir.format(cfg.output_file_model_info))
    
    return ("DONE",)           