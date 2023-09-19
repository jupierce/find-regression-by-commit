#!/bin/bash

./main.py
day="$(date +%F)"
hour="$(date +"%H-%M-%S")"

mkdir -p "${day}"
mv analysis "${day}/${hour}"
gsutil cp -r "${day}/${hour}"  "gs://origin-ci-test/mechanical-deads/${day}/"