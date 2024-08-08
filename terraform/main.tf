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

provider "google" {

}

provider "archive" {}


# Required API services
resource "google_project_service" "enable_iam_credentials" {
  project                    = var.PROJECT_ID
  service                    = "iamcredentials.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_appengine_api" {
  project                    = var.PROJECT_ID
  service                    = "appengine.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}

resource "google_project_service" "enable_cloudbuild_api" {
  project                    = var.PROJECT_ID
  service                    = "cloudbuild.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_cloudfunctions_api" {
  project                    = var.PROJECT_ID
  service                    = "cloudfunctions.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_cloudrunadmin_api" {
  project                    = var.PROJECT_ID
  service                    = "run.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_servicemanagement_api" {
  project                    = var.PROJECT_ID
  service                    = "servicemanagement.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_serviceusage_api" {
  project                    = var.PROJECT_ID
  service                    = "serviceusage.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_servicecontrol_api" {
  project                    = var.PROJECT_ID
  service                    = "servicecontrol.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_gads_api" {
  project                    = var.PROJECT_ID
  service                    = "googleads.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}
resource "google_project_service" "enable_vertexai_api" {
  project                    = var.PROJECT_ID
  service                    = "aiplatform.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Service Account definition
resource "google_service_account" "sa" {
  project      = var.PROJECT_ID
  account_id   = var.SERVICE_ACCOUNT
  display_name = "Service Account for running copycat"
}
resource "google_project_iam_member" "editor-sa" {
  project = var.PROJECT_ID
  role    = "roles/editor"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "service-agent" {
  project = var.PROJECT_ID
  role    = "roles/run.serviceAgent"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "cloud-run-admin" {
  project = var.PROJECT_ID
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "serviceusage-admin" {
  project = var.PROJECT_ID
  role    = "roles/serviceusage.serviceUsageAdmin"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "eventarc-admin" {
  project = var.PROJECT_ID
  role    = "roles/eventarc.admin"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "token-creator" {
  project = var.PROJECT_ID
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_binding" "users-token-creator" {
  project = var.PROJECT_ID
  role    = "roles/iam.serviceAccountTokenCreator"
  members  = formatlist("%s%s", "user:", split(",", var.USER_LIST))
}
resource "google_project_iam_binding" "users-oidc-token-creator" {
  project = var.PROJECT_ID
  role    = "roles/iam.serviceAccountOpenIdTokenCreator"
  members  = formatlist("%s%s", "user:", split(",", var.USER_LIST))
}
resource "google_project_iam_member" "oidc-token-creator" {
  project = var.PROJECT_ID
  role    = "roles/iam.serviceAccountOpenIdTokenCreator"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "log-writer" {
  project  = var.PROJECT_ID
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "storage-admin" {
  project  = var.PROJECT_ID
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.sa.email}"
}
resource "google_project_iam_member" "artifact-registry-writer" {
  project  = var.PROJECT_ID
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.sa.email}"
}

resource "google_project_iam_member" "compute-storage-admin" {
  project  = var.PROJECT_ID
  role    = "roles/storage.admin"
  member  = "serviceAccount:${var.PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
}
resource "google_project_iam_member" "compute-log-writer" {
  project  = var.PROJECT_ID
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${var.PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
}


resource "google_project_iam_member" "event-receiving" {
  project  = var.PROJECT_ID
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.sa.email}"
}

resource "google_project_iam_member" "aiplatform-admin" {
  project  = var.PROJECT_ID
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.sa.email}"
}

resource "google_storage_bucket" "copycat_build_bucket" {
  project                     = var.PROJECT_ID
  name                        = "${var.PROJECT_ID}-${var.DEPLOYMENT_NAME}-${var.BUILD_GCS_BUCKET}"
  location                    = var.LOCATION
  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "Delete"
    }
  }
}


terraform {
  required_providers {
     google = {
      version = "~> 5.30.0"
    }
  } 
}