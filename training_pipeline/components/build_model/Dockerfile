# https://hub.docker.com/r/pytorch/torchserve/tags
FROM pytorch/torchserve:latest-gpu

# build variables
ARG _MODEL_NAME

RUN echo $_MODEL_NAME

# install dependencies
USER root
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
USER model-server

# copy model artifacts, custom handler and other dependencies
COPY ./custom_handler.py /home/model-server/
COPY ./model.pt /home/model-server/
COPY ./helpers.zip /home/model-server/
COPY ./label_map.json /home/model-server/

RUN echo `ls`

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
    --model-name=$_MODEL_NAME \
    --version=1.0 \
    --serialized-file=/home/model-server/model.pt \
    --handler=/home/model-server/custom_handler.py \
    --extra-files "/home/model-server/helpers.zip,/home/model-server/label_map.json" \
    --export-path=/home/model-server/model-store 

# run Torchserve HTTP serve to respond to prediction requests
# https://pytorch.org/serve/server.html
CMD ["torchserve", \
    "--start", \
    "--ts-config=/home/model-server/config.properties", \
    "--models", \
    "all", \
    "--model-store", \
    "/home/model-server/model-store"]
# alternative to 'all' specify the model directly like '$_MODEL_NAME=$_MODEL_NAME.mar'