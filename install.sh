#!/bin/bash

apt-get update
# apt install -y python3 python3-pip
apt install htop
pip install google-cloud-storage google-cloud-bigquery future lxml cython fast_fisher tqdm pandas db-dtypes google-cloud-bigquery-storage matplotlib
pip install -r requirements.txt
