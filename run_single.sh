#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing network (network)"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Missing optimizer (opt)"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Missing dataset (dts)"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Missing seed (seed)"
    exit 1
fi

docker run --rm -e argg="1" -v /home/lorenzo/Projects/OMML_rev_rev:/work/project/ --gpus all --ipc host -u 1001:1001 lorenzopapa5/cuda11.2-python3.8-tf2.8.0-omml /usr/bin/python3 /work/project/main_cifar_new.py --network "${1}" --opt "${2}" --dts "${3}" --seed "${4}"
