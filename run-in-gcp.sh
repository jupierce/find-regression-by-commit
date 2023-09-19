#!/bin/bash

gcloud compute instances create mechanical-deads-driver \
    --project=openshift-gce-devel \
    --zone=us-east1-c \
    --machine-type=n2-standard-96 \
    --network-interface=network=default,network-tier=PREMIUM,stack-type=IPV4_ONLY \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --service-account=aos-kettle@openshift-gce-devel.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=mechanical-deads-driver,image=projects/debian-cloud/global/images/debian-11-bullseye-v20230912,mode=rw,size=10,type=projects/openshift-gce-devel/zones/us-east1-c/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any

while true; do
  gcloud compute ssh --project=openshift-gce-devel --zone=us-east1-c mechanical-deads-driver -- echo ok
  if [[ "$?" == "0" ]]; then
    break
  fi
  echo "Waiting for instance..."
  sleep 20
done

gcloud compute scp --project=openshift-gce-devel --zone=us-east1-c requirements.txt mechanical-deads-driver:~/requirements.txt
gcloud compute scp --project=openshift-gce-devel --zone=us-east1-c install.sh mechanical-deads-driver:~/install.sh
gcloud compute scp --project=openshift-gce-devel --zone=us-east1-c main.py mechanical-deads-driver:~/main.py
gcloud compute scp --project=openshift-gce-devel --zone=us-east1-c run-main.sh mechanical-deads-driver:~/run-main.sh
gcloud compute ssh --project=openshift-gce-devel --zone=us-east1-c mechanical-deads-driver -- chmod +x ~/*.sh
gcloud compute ssh --project=openshift-gce-devel --zone=us-east1-c mechanical-deads-driver -- chmod +x ~/*.py

gcloud compute ssh --project=openshift-gce-devel --zone=us-east1-c mechanical-deads-driver -- ./install.sh
gcloud compute ssh --project=openshift-gce-devel --zone=us-east1-c mechanical-deads-driver -- ./run-main.sh

gcloud compute instances delete --project=openshift-gce-devel --zone=us-east1-c mechanical-deads-driver