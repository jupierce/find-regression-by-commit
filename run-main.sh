#!/bin/bash

if [[ "$1" == "" ]]; then
  echo "Release must be specified"
  echo "Syntax example: $0 4.14"
  exit 1
fi

release="$1"

./main.py --release="${release}"
day="$(date +%F)"
hour="$(date +"%H-%M-%S")"

mkdir -p "${day}/${release}"
mv analysis "${day}/${release}/${hour}"
gsutil cp -r "${day}/${release}/${hour}"  "gs://origin-ci-test/mechanical-deads/${day}/${release}/"

# If we are running in a google VM
if curl metadata.google.internal -i; then
  # Note: inside the container the name is exposed as $HOSTNAME
  INSTANCE_NAME=$(curl -sq "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
  INSTANCE_ZONE=$(curl -sq "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google")

  echo "Terminating instance [${INSTANCE_NAME}] in zone [${INSTANCE_ZONE}}"
  TOKEN=$(curl -sq "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" -H "Metadata-Flavor: Google" | jq -r '.access_token')
  curl -X DELETE -H "Authorization: Bearer ${TOKEN}" https://www.googleapis.com/compute/v1/$INSTANCE_ZONE/instances/$INSTANCE_NAME
fi