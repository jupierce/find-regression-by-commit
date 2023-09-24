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