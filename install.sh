#!/bin/bash

sudo apt-get update
sudo apt install -y python3 python3-pip htop
sudo pip install google-cloud-storage google-cloud-bigquery future lxml cython fast_fisher tqdm pandas db-dtypes google-cloud-bigquery-storage matplotlib
sudo pip install -r requirements.txt
