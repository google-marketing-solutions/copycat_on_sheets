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


resource "time_sleep" "wait_60s" {
  create_duration = "60s"

  depends_on = [
    google_project_iam_member.log-writer,
    google_project_iam_member.artifact-registry-writer,
    google_project_iam_member.storage-admin,
  ]
}

resource "google_cloud_run_v2_service_iam_binding" "runner_cf_cr_binding" {
  location = google_cloudfunctions2_function.runner.location
  project  = google_cloudfunctions2_function.runner.project
  name     = google_cloudfunctions2_function.runner.name
  role     = "roles/run.invoker"
  members        = formatlist("%s%s", "user:", split(",", var.USER_LIST))
}

resource "google_cloud_run_v2_service_iam_binding" "runner_cf_srva_binding" {
  location = google_cloudfunctions2_function.runner.location
  project  = google_cloudfunctions2_function.runner.project
  name     = google_cloudfunctions2_function.runner.name
  role     = "roles/cloudfunctions.serviceAgent"
  members = [
    "serviceAccount:${google_service_account.sa.email}",
  ]
}

data "archive_file" "runner_archive" {
  type        = "zip"
  output_path = ".temp/runner_code_source.zip"
  source_dir  = "${path.module}/../cloud_functions/runner/"

  depends_on = [google_storage_bucket.copycat_build_bucket]
}

resource "google_storage_bucket_object" "runner_object" {
  name       = "${var.DEPLOYMENT_NAME}-runner-${data.archive_file.runner_archive.output_sha256}.zip"
  bucket     = google_storage_bucket.copycat_build_bucket.name
  source     = data.archive_file.runner_archive.output_path
  depends_on = [data.archive_file.runner_archive]
}

resource "google_cloudfunctions2_function" "runner" {
  name        = "${var.DEPLOYMENT_NAME}-runner"
  description = "It runs a copycat execution receiving and input google sheet URL and the name of the sheet with the configuration"
  project     = var.PROJECT_ID
  location    = var.REGION
  depends_on = [ google_storage_bucket.copycat_build_bucket, google_storage_bucket_object.runner_object, time_sleep.wait_60s]

  build_config {
    runtime     = "python310"
    entry_point = "run" # Set the entry point
    service_account = google_service_account.sa.name
    environment_variables = {
      BUILD_CONFIG_TEST = "build_test"
    }
    source {
      storage_source {
        bucket = google_storage_bucket.copycat_build_bucket.name
        object = google_storage_bucket_object.runner_object.name
      }
    }
  }

  service_config {
    min_instance_count = 0
    available_cpu = 4
    available_memory   = "16Gi"
    timeout_seconds    = 3600
    environment_variables = {
      PROJECT_ID      = var.PROJECT_ID
      DEPLOYMENT_NAME = var.DEPLOYMENT_NAME
      SERVICE_ACCOUNT = google_service_account.sa.email
      REGION          = var.REGION
    }
    #ingress_settings               = "ALLOW_INTERNAL_ONLY"
    all_traffic_on_latest_revision = true
    service_account_email          = google_service_account.sa.email
  }
  lifecycle {
    ignore_changes = [
      # Ignore changes to generation
      build_config[0].source[0].storage_source[0].generation
    ]
  }
}

output PROJECT_NUMBER {
  value =  var.PROJECT_NUMBER
  depends_on = [ google_cloudfunctions2_function.runner ]
}

output COPYCAT_SERVICE_ACCOUNT {
  value = google_service_account.sa.email
  depends_on = [ google_cloudfunctions2_function.runner ]
}

output COPYCAT_ENDPOINT_URL {
  value = google_cloudfunctions2_function.runner.url
  depends_on = [ google_cloudfunctions2_function.runner ]
}
