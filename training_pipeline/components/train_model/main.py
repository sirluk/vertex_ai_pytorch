from typing import NamedTuple


def train_model(
    comp_params: dict
) -> NamedTuple('train_model_outputs', [
    ('DUMMY', str)
]):
    
    import argparse
    from google.cloud import aiplatform
    
    cfg = argparse.Namespace(**comp_params)
        
    # trainer image
    trainer_image_uri = f'{cfg.build_location}-docker.pkg.dev/{cfg.project_id}/{cfg.trainer_image_name}/{cfg.trainer_image_name}'
    
    trainer_args = [
        f'--base-dir={cfg.working_dir}',
        f'--model-dir={cfg.working_dir}/{cfg.model_dir}',
        f'--training-data-uri={cfg.bucket}/train_data/{cfg.language}/ds_train.csv',
        f'--validation-data-uri={cfg.bucket}/train_data/{cfg.language}/ds_validation.csv',
        '--data-format=gcs',
        f'--language={cfg.language}',
        f'--model_name={cfg.model_name}',
        f'--model_name_output={cfg.model_name_output}',
        f'--lr={cfg.lr}',
        f'--dropout={cfg.dropout}',
        f'--n_hidden={cfg.n_hidden}',
        f'--batch_size={cfg.batch_size}',
        f'--num_epochs={cfg.num_epochs}',
        f'--val_size={cfg.val_size}',
        f'--output_file_label_map={cfg.output_file_label_map}'
    ]
    
    try:
        debug = eval(cfg.debug.lower())
    except (NameError, AttributeError) as e:
        print(repr(e))
        debug = False   
    if debug: trainer_args.append('--debug')
    
    tensorboard_resource_name = f'projects/{cfg.project_id}/locations/{cfg.model_location}/tensorboards/{cfg.tensorboard_id}'

    aiplatform.init(project=cfg.project_id, location=cfg.model_location, staging_bucket=cfg.bucket)
    
    display_name_parts = [
        cfg.dt,
        cfg.language,
        cfg.model_name,
        cfg.batch_size,
        cfg.lr
    ]
    job = aiplatform.CustomContainerTrainingJob(
        display_name = "_".join([str(x) for x in display_name_parts]),
        container_uri = trainer_image_uri + ':latest'
    )

    model = job.run(
        args = trainer_args,
        base_output_dir = cfg.working_dir,
        replica_count = int(cfg.trainer_replica_count),
        machine_type = cfg.trainer_machine_type,
        accelerator_type = cfg.trainer_gpu_type,
        accelerator_count = int(cfg.trainer_n_gpu),
        tensorboard = tensorboard_resource_name,
        service_account = cfg.service_account,
        sync = True
    )
    
    return ("DONE",)