#!/bin/bash
# Tests the initial call
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CONFIG_PATH="../../terraform/terraform.tfvars"

set -a
eval "$(cat ${CONFIG_PATH} | sed -e 's/ *= */=/g')"

curl localhost:8080   -X POST   -H "Content-Type: application/json"   -H "ce-id: 123451234512345"   -H "ce-specversion: 1.0"   -H "ce-time: 2020-01-02T12:34:56.789Z"  -H "x-cloudtasks-taskretrycount: 0"   -d '{"worksheet_url": "https://docs.google.com/spreadsheets/d/1Bpcj4rC3Ytc5mQVuuvP7L1cD4J1xa9GwEnYnUwJyKgA",
    "config_sheet_name": "config"
}'


