from kfp import dsl
from kfp import components as comp

from .components.train_model.main import train_model
from .components.build_model.main import build_model
from .components.deploy_model.main import deploy_model
from .components.save_model_info.main import save_model_info


train_model_op_fun = (
    comp.create_component_from_func(
        train_model,
        packages_to_install=[
            "kfp==1.8.18",
            "fire==0.4.0",
            "gcsfs==2022.11.0",
            "google-cloud-aiplatform==1.20.0"
        ],
        base_image="python:3.9.16"
    )
)

build_model_op_fun = (
    comp.create_component_from_func(
        build_model,
        packages_to_install=[
            "kfp==1.8.18",
            "google-cloud-build==3.10.0"
        ],
        base_image="python:3.9.16"
    )
)

deploy_model_op_fun = (
    comp.create_component_from_func(
        deploy_model,
        packages_to_install=[
            "kfp==1.8.18",
            "google-cloud-aiplatform==1.20.0"
        ],
        base_image="python:3.9.16"
    )
)

save_model_info_op_fun = (
    comp.create_component_from_func(
        save_model_info,
        packages_to_install=[
            "kfp==1.8.18",
            "google-cloud-storage==2.7.0" #1.44.0
        ],
        base_image="python:3.9.16"
    )
)


@dsl.pipeline(
    name='vertex-training-pipeline',
    description='vertex-training-pipeline'
)
def training_pipeline(
    comp_params: dict,
    model_info: dict
):
    
    train_model_task = train_model_op_fun(
        comp_params
    ) \
    .set_cpu_limit('500m') \
    .set_memory_limit('2G') \
    .set_display_name('Train Model')

    
    build_model_task = build_model_op_fun(
        comp_params
    ) \
    .set_cpu_limit('500m') \
    .set_memory_limit('2G') \
    .set_display_name('Build Model') \
    .after(train_model_task)
    
    
    deploy_model_task = deploy_model_op_fun(
        comp_params,
        endpoint_image_uri = build_model_task.outputs["endpoint_image_uri"],
        commit_sha = build_model_task.outputs["commit_sha"]
    ) \
    .set_cpu_limit('500m') \
    .set_memory_limit('2G') \
    .set_display_name('Deploy Model') \
    .after(build_model_task)
    
    
    save_model_info_task = save_model_info_op_fun(
        comp_params,
        model_info,
        model_resource_name = deploy_model_task.outputs["model_resource_name"]
    ) \
    .set_cpu_limit('500m') \
    .set_memory_limit('2G') \
    .set_display_name('Save Model Info') \
    .after(deploy_model_task)