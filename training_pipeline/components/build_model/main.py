from typing import NamedTuple


def build_model(
    comp_params: dict,
) -> NamedTuple('build_model_outputs', [
    ('endpoint_image_uri', str),
    ('commit_sha', str)
]):

    import os
    import argparse
    from google.cloud.devtools import cloudbuild
    
    cfg = argparse.Namespace(**comp_params)
    
    endpoint_image_uri = f"{cfg.build_location}-docker.pkg.dev/{cfg.project_id}/{cfg.endpoint_image_name}/{cfg.endpoint_image_name}_{cfg.language}"

    client = cloudbuild.CloudBuildClient()
    
    request = cloudbuild.RunBuildTriggerRequest(
        project_id = cfg.project_id,
        trigger_id = cfg.trigger_name,
        # https://cloud.google.com/python/docs/reference/cloudbuild/latest/google.cloud.devtools.cloudbuild_v1.types.RepoSource
        source = {
            "branch_name": cfg.repo_branch,
            "dir_": cfg.build_dir,
            "substitutions": {
                "_IMG_URI": endpoint_image_uri,
                "_MODEL_DIR": f"{cfg.working_dir}/{cfg.model_dir}",
                "_TRAINER_CODE_DIR": cfg.trainer_code_dir,
                "_BUILD_DIR": cfg.build_dir,
                "_MODEL_NAME": cfg.endpoint_model_name,
                "_LABEL_MAP_FILE": cfg.output_file_label_map
            }
        }
    )

    operation = client.run_build_trigger(request=request)
    
    response = operation.result()
    
    return (endpoint_image_uri, response.substitutions["COMMIT_SHA"])