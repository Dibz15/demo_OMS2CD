#!/bin/bash

docker run --rm \
    -v $(pwd)/OMS2CD/:/data/ \
    -v $(pwd)/outputs/:/outputs/ \
    -it \
    -p 8888:8889 \
    oms2cd:latest \
    jupyter notebook --no-browser --allow-root --port=8889 --ip 0.0.0.0 --NotebookApp.allow_origin='*'