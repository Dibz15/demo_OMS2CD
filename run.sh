#!/bin/bash

docker run --rm\
    -v $(pwd):/src/ \
    -v $(pwd)/OMS2CD/:/data/ \
    -v $(pwd)/outputs/:/outputs/ \
     dibz15/oms2cd "$@"
