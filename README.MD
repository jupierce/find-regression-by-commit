
- Setup massive system in GCE running debian (e.g. n2-highcpu-96) - configure to runs as aos-kettle.
- sudo apt install python3 python3-pip htop
- sudo pip install google-cloud-storage google-cloud-bigquery future lxml cython fast_fisher tqdm pandas db-dtypes google-cloud-bigquery-storage matplotlib
- Upload main.py
- chmod +x main.py
- ulimit -n 5000