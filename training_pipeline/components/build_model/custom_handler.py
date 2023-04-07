
import zipfile
with zipfile.ZipFile("helpers.zip","r") as zip_ref:
    zip_ref.extractall(".")

import json
import os
import logging
import torch
from ts.torch_handler.base_handler import BaseHandler
from transformers import BertTokenizer

from os import listdir
from os.path import isfile, join

from helpers.model import BertClf


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False
        self.manifest = None
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Loads the model.bin file and initializes the model object.
        Instantiates Tokenizer for preprocessor to use
        Modify the self.model = ... line to load the model in a
        correct manner.
        """
        
        # Parse context and set attributes
        self.manifest = context.manifest
        serialized_file = self.manifest["model"]["serializedFile"]
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Set device to GPU if cuda available else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Find .bin or .pt file
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")

        # Load model
        self.model = BertClf.load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model.model_name)
        
        # load mapping file
        with open("label_map.json", "r") as f:
            self.label_map = {int(k):v for k,v in json.load(f).items()}

        self.initialized = True


    def preprocess(self, data):
        """
        Transform raw input into model input data.
        Args:
            batch: list of raw requests, should match batch size
        Returns:
            list of preprocessed model input data
        """
        print(f"input data:\n{data}")
        
        # TODO build in max batch size

        texts = []
        for i,d in enumerate(data):
            texts.append(d.get("text"))
        
        text_inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        
        return text_inputs


    def inference(self, text_inputs):
        """
        Internal inference methods
        Args:
            model_input: preprocessed model input data
        Returns:
            list of inference output in NDArray
        """
        
        with torch.no_grad():
            return self.model(
                input_ids=text_inputs['input_ids'].to(self.device)
            ).cpu()

  
    def postprocess(self, model_output, data):
        """
        Return postprocessed inference result.
        Args:
            inference_output: list of inference output
        Returns:
            list of postprocessed results
        """
        prediction = [[self.label_map[i] for i, logit in enumerate(logits) if logit>0] for logits in model_output]
        
        processed_prediction = []
        for i,p in enumerate(prediction):
            d = {
                "pred": p,
                "pred_raw": model_output[i].tolist(),
                "id": data[i].get("id")
            }
            processed_prediction.append(d)
        
        return processed_prediction


    def handle(self, data, context):
        """
        Invoked by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        Note that this function can be omitted.
        Args:
            data: Input data for prediction
            context: Initial context contains model server system properties.
        Returns:
            prediction output
        """
        text_inputs = self.preprocess(data)
        model_output = self.inference(text_inputs)
        return self.postprocess(model_output, data)