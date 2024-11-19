# Copycat on sheets

## Problem

Automatically create ad copies at scale based on brand style by Generative AI

## Solution

Copycat on sheets leverages the [Copycat library](https://github.com/google-marketing-solutions/copycat/blob/main/README.md) power withing Google Sheets.

Copycat is a Python package that leverages Google Gemini models to generate high-quality ad copies for Google Search campaigns. Whether you need Responsive Search Ads or Text Ads, Copycat learns from your top-performing ads and brand guidelines to create copy that seamlessly aligns with your brand voice.

To do so, Copycat on Sheets, deploys a Cloud Run Function in your Google Cloud a project that provides the processing endpoint to be called by the authorized users from the ad-hoc Copycat menu created on the Google Sheet.

## Installation Steps

1. `git clone` this repository
2. Navigate to the directory `copycat_on_sheets/terraform`
3. Edit the file `terraform.tfvars` replacing the values for your project:

   * `PROJECT_ID` = is the name of the Google Cloud project where you will deploy. i.e: "my-project"
   * `PROJECT_NUMBER` = is the number of the Google Cloud project. It's different from the name). i.e: "8888888"
   * `REGION` = the Google Cloud region where you will deploy. i.e: "us-central1"
   * `LOCATION` = the broader region where you will be deploying. i.e: "US"
   * `DEPLOYMENT_NAME` = The prefix to append to all the created resources so they can be easily identified. i.e:"copycat"
   * `SERVICE_ACCOUNT` = The name of the service account to be created. i.e:"copycat-sa"
   * `USER_LIST` = A list of emails separated by comma, representing the persons with access to the tool. The emails must be accounts granted into the Google Cloud Project. i.e: "<myemail@somecompany.com>,<youremail@somecompany.com>"

4. Open a shell terminal and navigate to `copycat_on_sheets/terraform` directory. Execute the command: `terraform apply`
5. If prompted to confirm the deployment type: `yes`
6. When everything completed successfully, make sure you write down the information displayed in the console at the end of the process, i.e:

    * `PROJECT_NUMBER=8888888`
    * `COPYCAT_SERVICE_ACCOUNT=copycat-sa@my-project.iam.gserviceaccount.com`
    * `COPYCAT_ENDPOINT_URL=https://us-central1-my-project.cloudfunctions.net/copycat-runner`

7. Make a copy of [this Google Spreadsheet](https://docs.google.com/spreadsheets/d/1-TKu0WJcASacnTs2CoKujJ3gAvw5EhoDTx-kEtadHsA/edit?usp=sharing), it will be used to run copycat.
8. Open the sheet and grant edit permissions to the service account noted in step 6 and all the members of the team who will be using the tool, defined on the `USER_LIST` of Step 3. 
9. Open the Google Sheet, navigate to the `config` tab and replace the value of the following variables obtained during the deployment process:

    * `COPYCAT_ENDPOINT_URL`
    * `COPYCAT_SERVICE_ACCOUNT`

10. On the menu bar on the top, click on `Extension` > `Apps Script`
11. On the new window, on the left pane menu, click on `Project Settings` and on the section "Google Cloud Platform (GCP) Project", click on `Change Project` and add the value of `PROJECT_NUMBER` from Step 6.
12. You are all set!! Follow the instructions on the `instruction` tab of the sheet.
