gsutil -m cp -r $_MODEL_DIR/* $_BUILD_DIR/.
mv `ls $_BUILD_DIR/*.pt | head -1` $_BUILD_DIR/model.pt

# incase $_LABEL_MAP_FILE != label_map.json
mv $_BUILD_DIR/$_LABEL_MAP_FILE $_BUILD_DIR/label_map.json

cp -r $_TRAINER_CODE_DIR/helpers $_BUILD_DIR/helpers

# in case we want to copy the file back to the parent directory we can use the following_
# pushd $_BUILD_DIR; zip -r helpers.zip helpers; cp helpers.zip "$(dirs -l +1)"; popd
pushd $_BUILD_DIR; zip -r helpers.zip helpers; popd

# replace string in Dockerfile
# sed -i "s/LABEL_MAP_FILE/$_LABEL_MAP_FILE/g" Dockerfile

echo `ls $_BUILD_DIR`

echo "copied artifacts to build dir"
