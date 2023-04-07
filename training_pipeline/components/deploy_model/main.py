from typing import NamedTuple


def deploy_model(
    comp_params: dict,
    endpoint_image_uri: str,
    commit_sha: str
) -> NamedTuple('deploy_model_outputs', [
    ('model_resource_name', str)
]):
    
    import argparse
    from google.cloud import aiplatform
    
    cfg = argparse.Namespace(**comp_params)
    
    aiplatform.init(project=cfg.project_id, location=cfg.model_location)
    
    model_kwargs = {
        "serving_container_image_uri": endpoint_image_uri + ":" + commit_sha,
        "serving_container_predict_route": f"{cfg.endpoint_predict_route}/{cfg.endpoint_model_name}",
        "serving_container_health_route": cfg.endpoint_health_route,
        "serving_container_ports": [7080]
    }

    try:
        parent_model = [x for x in aiplatform.Model.list() if x.display_name == cfg.endpoint_model_name][0]
        model_kwargs["parent_model"] = parent_model.resource_name
    except IndexError:
        model_kwargs["display_name"] = cfg.endpoint_model_name

    model = aiplatform.Model.upload(**model_kwargs)

    endpoint = aiplatform.Endpoint.create(
        display_name = cfg.endpoint_model_name,
        project = cfg.project_id,
        location = cfg.model_location
    )

    model.deploy(
        endpoint = endpoint,
        deployed_model_display_name = cfg.endpoint_model_name,
        machine_type = cfg.endpoint_machine_type
    )

    return (model.resource_name,)