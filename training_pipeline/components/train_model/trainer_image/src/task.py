import os
import argparse
import model_trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model_dir',
                        default=os.environ['AIP_MODEL_DIR'], type=str, help='Model dir.')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir',
                        default=os.environ['AIP_CHECKPOINT_DIR'], type=str, help='Checkpoint dir set during Vertex AI training.')    
    parser.add_argument('--tensorboard-dir', dest='tensorboard_dir',
                        default=os.environ['AIP_TENSORBOARD_LOG_DIR'], type=str, help='Tensorboard dir set during Vertex AI training.') 
    parser.add_argument('--data-format', dest='data_format', type=str, help="Tabular data format set during Vertex AI training. E.g.'csv', 'bigquery'")
    
    parser.add_argument('--training-data-uri', dest='training_data_uri', type=str, help='Training data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--validation-data-uri', dest='validation_data_uri', type=str, help='Validation data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--test-data-uri', dest='test_data_uri', type=str, help='Test data GCS or BQ URI set during Vertex AI training.')
    
    # Custom Args
    parser.add_argument('--base-dir', dest='base_dir', type=str, help='Base dir for artefacts.')
    parser.add_argument('--language', dest='language', type=str, help='language for which to do training run.')
    parser.add_argument('--model_name', dest='model_name', type=str, help='name of a pretrained transformers model')
    parser.add_argument('--model_name_output', dest='model_name_output', type=str, help='name of a pretrained transformers model')
    parser.add_argument('--output_file_label_map', dest='output_file_label_map', type=str, help='output filename for label mapping')
    parser.add_argument('--output_file_tp_map', dest='output_file_tp_map', type=str, help='output filename for touchpoint mapping')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='Learning rate for optimizer.')
    parser.add_argument('--dropout', dest='dropout', default=0.2, type=float, help='Float percentage of DNN nodes [0,1] to drop for regularization.')   
    parser.add_argument('--n_hidden', dest='n_hidden', default=1, type=int, help='Number of hidden layers in head')   
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='Number of examples during each training iteration.')    
    parser.add_argument('--num_epochs', dest='num_epochs', default=20, type=int, help='Number of passes through the dataset during training to achieve convergence.')
    parser.add_argument('--val_size', dest='val_size', default=0.1, type=float, help='Validation dataset share')
    parser.add_argument('--debug', action='store_true', help='Whether to run on small subset for testing')
    
    hparams = parser.parse_args()   
    print(hparams)
    model_trainer.train_model(hparams)
