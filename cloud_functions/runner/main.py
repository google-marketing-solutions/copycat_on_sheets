"""Google Cloud function that uploads the chunk of conversions to Google Ads."""

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
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import Any, Dict, Union
import flask
import functions_framework
import google.auth
from google.cloud import logging
import gspread
import numpy as np
import pandas as pd

sys.path.insert(0, "./lib")
from copycat.py import copycat

sheet_client = None

DEFAULT_CONFIG = {
    "COMPANY_NAME": "My Company",
    "AD_REPORT_URL": "",
    "AD_REPORT_SHEET_NAME": "Ad Report",
    "SEARCH_KEYWORD_REPORT_URL": "",
    "SEARCH_KEYWORD_REPORT_SHEET_NAME": "",
    "NEW_KEYWORDS_URL": "",
    "NEW_KEYWORDS_SHEET_NAME": "",
    "USE_STYLE_GUIDE": "",
    "EMBEDDING_MODEL_NAME": "text-embedding-ada-002",
    "AD_FORMAT": "RSA",
    "LANGUAGE": "English",
    "NUM_IN_CONTEXT_EXAMPLES": 100,
    "CHAT_MODEL_NAME": "gemini-1.5-pro-preview-0514",
    "TEMPERATURE": 0.7,
    "TOP_K": 50,
    "TOP_P": 1,
    "BATCH_SIZE": 10,
    "DATA_LIMIT": 0,
}

FAILED_STATUS_FOR_REPORTING = "ERROR_IN_COPYCAT_RUNNER"
CF_NAME = "copycat_runner"


@functions_framework.http
def run(request: flask.Request) -> flask.Response:
  """HTTP Cloud Function.

  Args:
      request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>

  Returns:
      The response text, or any set of values that can be turned into a
      Response object using `make_response`
      <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
  """

  logging_client = logging.Client()
  log_name = os.environ["DEPLOYMENT_NAME"] + CF_NAME
  logger = logging_client.logger(log_name)

  required_elem = [
      "PROJECT_ID",
      "REGION",
      "SERVICE_ACCOUNT",
      "DEPLOYMENT_NAME",
  ]
  if not all(elem in os.environ for elem in required_elem):
    logger.log_text(
        f"{FAILED_STATUS_FOR_REPORTING}: Cannot proceed, there are missing"
        " input values please make sure you set all the environment variables"
        " correctly."
    )

    sys.exit(1)

  logger.log(request)
  request_json = request.get_json(silent=True)
  logger.log(request_json)

  try:

    worksheet_url = request_json["worksheet_url"]
    config_sheet_name = request_json["config_sheet_name"]
    sheets_client = _init_google_sheet_client()
    config = _read_config_from_google_sheet(
        DEFAULT_CONFIG, sheets_client, worksheet_url, config_sheet_name
    )

    (
        ad_report_data,
        keywords_data,
        new_keywords_data,
    ) = _load_data(sheets_client, config)

    (training_data, test_data) = _prepare_training_data(
        ad_report_data, keywords_data, new_keywords_data
    )
    training_data = (
        training_data[: int(config["DATA_LIMIT"])]
        if int(config["DATA_LIMIT"]) > 0
        else training_data
    )
    model = copycat.Copycat.create_from_pandas(
        training_data=training_data,
        embedding_model_name=config["EMBEDDING_MODEL_NAME"],
        persist_path="/tmp",
        ad_format=config["AD_FORMAT"],
    )

    style_guide = _generate_style_guide(config, training_data)

    generated_ads = []

    n_batches = np.ceil(len(test_data["keywords"]) / int(config["BATCH_SIZE"]))

    for data_batch in np.array_split(test_data, n_batches, axis=0):
      generated_ads.extend(
          model.generate_new_ad_copy(
              keywords=data_batch["keywords"].values.tolist(),
              keywords_specific_instructions=data_batch[
                  "information"
              ].values.tolist(),
              system_instruction_kwargs=dict(
                  company_name=config["COMPANY_NAME"],
                  language=config["LANGUAGE"],
              ),
              style_guide=style_guide,
              num_in_context_examples=int(config["NUM_IN_CONTEXT_EXAMPLES"]),
              model_name=config["CHAT_MODEL_NAME"],
              temperature=float(config["TEMPERATURE"]),
              top_k=float(config["TOP_K"]),
              top_p=float(config["TOP_P"]),
          )
      )

      test_data["generated_ad_object"] = pd.Series(
          generated_ads, index=data_batch["keywords"].index
      )

    results = _extract_resulting_ads(test_data)

    _write_result_into_google_sheet(config, sheets_client, results)

    return ("Results generated", 200)
  except Exception as e:
    logger.log(str(e))
    logging_payload = {
        "worksheet_url": worksheet_url if worksheet_url else None,
        "config_sheet_name": config_sheet_name if config_sheet_name else None,
        "status": FAILED_STATUS_FOR_REPORTING,
        "message": (
            str(e)
            if len(str(e) > 1)
            else "Check you shared the spreadsheet with the service account"
        ),
        "success": False,
    }

    logger.log_text(
        f"{FAILED_STATUS_FOR_REPORTING}: {json.dumps(logging_payload)}"
    )

    return (f"Error found: {str(e)}", 500)


def _extract_descriptions(data: pd.Series) -> pd.Series:
  """Extracts descriptions from a Pandas Series of Google Ad objects.

  Args:
    data: A Pandas Series where each element is assumed to have a 'google_ad'
      attribute containing descriptions.

  Returns:
    A Pandas Series of descriptions, with the index formatted as 'Description
    1',
    'Description 2', etc.
  """

  return pd.Series(
      data.google_ad.descriptions,
      index=[
          f"Description {i+1}" for i in range(len(data.google_ad.descriptions))
      ],
  )


def _extract_headlines(data: pd.Series) -> pd.Series:
  """Extracts headlines from a Pandas Series of Google Ad objects.

  Args:
    data: A Pandas Series where each element is assumed to have a 'google_ad'
      attribute containing headlines.

  Returns:
    A Pandas Series of headlines, with the index formatted as 'Headline 1',
    'Headline 2', etc.
  """
  return pd.Series(
      data.google_ad.headlines,
      index=[f"Headline {i+1}" for i in range(len(data.google_ad.headlines))],
  )


def _extract_resulting_ads(data: pd.Series) -> pd.Series:
  """Extracts headlines and descriptions from generated ad objects.

  This function takes a Pandas Series containing generated ad objects,
  extracts the headlines and descriptions from those objects, and then
  combines this information with the original keywords into a new DataFrame.

  Args:
    data: A Pandas Series where each element is a generated ad object.

  Returns:
    A Pandas Series containing the keywords, extracted headlines,
    and extracted descriptions.
  """
  headlines = data["generated_ad_object"].apply(_extract_headlines).fillna("--")
  descriptions = (
      data["generated_ad_object"].apply(_extract_descriptions).fillna("--")
  )

  return (
      data[["keywords"]]
      .copy()
      .merge(headlines, left_index=True, right_index=True)
      .merge(descriptions, left_index=True, right_index=True)
      .reset_index()
  )


def _load_data(
    sheets_client: gspread.client, config: Dict[str, Any]
) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Loads data from Google Sheets and prepares it for training.

  Args:
    sheets_client: gspread client object.
    config: A dictionary containing the configuration parameters.

  Returns:
    A tuple containing three pandas DataFrames:
      - clean_ad_report_data: Cleaned ad report data.
      - keywords_data: Keywords data.
      - new_keywords_data: New keywords data.
  """

  ad_report_data = _load_data_from_google_sheet(
      config["AD_REPORT_URL"],
      config["AD_REPORT_SHEET_NAME"],
      sheets_client,
      skip_rows=0,
  )

  search_keyword_report_data = _load_data_from_google_sheet(
      config["SEARCH_KEYWORD_REPORT_URL"],
      config["SEARCH_KEYWORD_REPORT_SHEET_NAME"],
      sheets_client,
      skip_rows=0,
  )

  new_keywords_data = _load_data_from_google_sheet(
      config["NEW_KEYWORDS_URL"],
      config["NEW_KEYWORDS_SHEET_NAME"],
      sheets_client,
      skip_rows=0,
  )

  clean_ad_report_data = _clean_ad_report_data(ad_report_data)

  keywords_data = search_keyword_report_data[
      ["Campaign ID", "Ad Group ID", "Keywords"]
  ].copy()

  keywords_data["Keyword"] = keywords_data["Keywords"].str.translate(
      {ord(i): None for i in '+[]"'}
  )

  keywords_data = keywords_data.dropna().drop_duplicates()

  keywords_data = keywords_data.groupby(["Campaign ID", "Ad Group ID"]).agg(
      keywords=("Keyword", lambda x: ", ".join(x).lower())
  )

  keywords_data = keywords_data.loc[keywords_data["keywords"] != ""]

  return (clean_ad_report_data, keywords_data, new_keywords_data)


def _prepare_training_data(
    ad_report_data: pd.DataFrame,
    keywords_data: pd.DataFrame,
    new_keywords_data: pd.DataFrame,
) -> Union[pd.DataFrame, pd.DataFrame]:
  """Prepares training and test data for the Copycat model.

  This function merges and processes the input data to create two DataFrames:
    - training_data: Contains keywords, headlines, and descriptions from
      existing ad campaigns.
    - test_data: Contains new keywords and any additional information
      provided for generating new ad copy.

  Args:
    ad_report_data: DataFrame containing ad performance data, including
      headlines and descriptions.
    keywords_data: DataFrame containing keywords grouped by campaign and ad
      group.
    new_keywords_data: DataFrame containing new keywords and optional additional
      information.

  Returns:
    A tuple containing two pandas DataFrames:
      - training_data: Prepared data for training the Copycat model.
      - test_data: Prepared data for generating new ad copy.
  """
  training_data = pd.merge(
      keywords_data,
      ad_report_data,
      how="inner",
      left_index=True,
      right_index=True,
  )

  test_data = new_keywords_data.groupby(["Campaign", "Ad Group"]).agg(
      keywords=("Keyword", lambda x: ", ".join(x).lower()),
      information=("Extra info", lambda x: ", ".join(x).lower()),
  )

  return (training_data, test_data)


def _generate_style_guide(
    config: Dict[str, Any], training_data: pd.DataFrame
) -> str:
  """Generates a style guide for ad copy based on existing ad data.

  This function uses a large language model (defined in the 'config') to
  analyze a DataFrame of existing ad copy ('training_data') and generate
  a style guide that captures the brand's tone, key phrases, and messaging
  patterns.

  Args:
    config: A dictionary containing configuration parameters, including: -
      'COMPANY_NAME': The name of the company. - 'USE_STYLE_GUIDE': A flag
      indicating whether to generate a style guide. - 'CHAT_MODEL_NAME': The
      name of the language model to use for generation. - 'TEMPERATURE': The
      temperature parameter for the language model. - 'TOP_K': The top_k
      parameter for the language model. - 'TOP_P': The top_p parameter for the
      language model.
    training_data: A DataFrame containing existing ad copy data, including
      columns for keywords, headlines, and descriptions.

  Returns:
    A string containing the generated style guide, or None if
    'USE_STYLE_GUIDE' is False.
  """

  if config["USE_STYLE_GUIDE"]:
    style_guide_prompt = """
    Below is an ad report for {company_name}, containing their ads (headlines
    and descriptions)
    that they use on Google Search Ads for the corresponding keywords.
    Headlines and descriptions are lists, and Google constructs ads by combining
    those headlines and descriptions together into ads. Therefore the headlines
    and descriptions should be sufficiently varied that Google is able to try
    lots of different combinations in order to find what works best.

    Use the ad report to write a comprehensive style guide for this brand's
    ad copies that can serve as instruction for a copywriter to write new
    ad copies for {company_name} for new lists of keywords.

    Ensure that you capure strong phrases, slogans and brand names
    of {company_name} in the guide.
    \n\n{style_guide}
    """.format(
        company_name=config["COMPANY_NAME"],
        style_guide=training_data.reset_index(drop=True).astype(str).to_csv(),
    )

    response = copycat.ad_copy_generator.generative_models.GenerativeModel(
        config["CHAT_MODEL_NAME"],
        generation_config={
            "temperature": float(config["TEMPERATURE"]),
            "top_k": float(config["TOP_K"]),
            "top_p": float(config["TOP_P"]),
        },
    ).generate_content(style_guide_prompt)

    return response.candidates[0].content.parts[0].text


def _init_google_sheet_client():
  """Initializes and authenticates a gspread client.

  This function sets up the necessary credentials and authorization
  to interact with Google Sheets using the gspread library.

  Returns:
    A gspread.Client object authorized to access Google Sheets.
  """
  scopes = [
      "https://www.googleapis.com/auth/spreadsheets",
      "https://www.googleapis.com/auth/drive",
      "https://spreadsheets.google.com/feeds",
  ]

  credentials, _ = google.auth.default(scopes=scopes)
  client = gspread.authorize(credentials)

  return client


def _read_config_from_google_sheet(
    default_config: Dict[str, str],
    sheets_client: gspread.client,
    worksheet_url: str,
    config_sheet_name: str,
) -> Dict[str, str]:
  """Reads the configuration parameters from a Google Sheet.

  Args:
    default_config: A dictionary containing the default configuration
      parameters.
    sheets_client: gspread client object.
    worksheet_url: The URL of the Google Sheet containing the configuration.
    config_sheet_name: The name of the sheet in the Google Sheet containing the
      configuration.

  Returns:
    A dictionary containing the configuration parameters.
  """
  data = _load_data_from_google_sheet(
      worksheet_url, config_sheet_name, sheets_client, skip_rows=0
  )

  for i, k in enumerate(data.VARIABLE):
    default_config[k] = data.VALUE[i]

  return default_config


def _load_data_from_google_sheet(
    url: str,
    worksheet_name: str,
    sheets_client: gspread.Client,
    skip_rows: int = 0,
) -> pd.DataFrame:
  """Loads data from a Google Sheet to pandas dataframe.

  Args:
    url: The URL of the Google Sheet.
    worksheet_name: The name of the worksheet to load data from.
    sheets_client: A gspread.Client object.
    skip_rows: The number of rows to skip from the beginning of the sheet.

  Returns:
    A pandas DataFrame containing the data from the specified Google Sheet.

  Raises:
    Exception: If there is an error loading data from the Google Sheet.
  """
  try:
    input_sheet = sheets_client.open_by_url(url)

    if worksheet_name:
      values = input_sheet.worksheet(worksheet_name).get_all_values()
    else:
      values = input_sheet.sheet1.get_all_values()

    return pd.DataFrame.from_records(
        values[skip_rows + 1 :], columns=values[skip_rows]
    )
  except Exception as e:
    print(e)
    raise (e)


def _clean_ad_report_data(ad_report_data: pd.DataFrame) -> pd.DataFrame:
  """Cleans and structures ad report data from a DataFrame.

  This function processes a DataFrame containing raw ad report data,
  extracting and structuring headlines and descriptions into lists
  associated with each campaign and ad group.

  Args:
    ad_report_data: A DataFrame containing raw ad report data, likely with
      columns for headlines, descriptions, and campaign/ad group IDs.

  Returns:
    A cleaned DataFrame with headlines and descriptions grouped as lists
    for each unique combination of "Campaign ID" and "Ad Group ID".
  """

  clean_ad_report_data = ad_report_data.copy()

  headline_cols = [
      c
      for c in clean_ad_report_data.columns
      if c.startswith("Headline") and not c.endswith("position")
  ]
  description_cols = [
      c
      for c in clean_ad_report_data.columns
      if c.startswith("Description") and not c.endswith("position")
  ]
  clean_ad_report_data["headlines"] = pd.Series(
      {
          k: list(
              filter(lambda x: x != "--" and x and not x.startswith("{"), v),
          )
          for k, v in clean_ad_report_data[headline_cols]
          .T.to_dict("list")
          .items()
      },
      index=clean_ad_report_data.index,
  )
  clean_ad_report_data["descriptions"] = pd.Series(
      {
          k: list(filter(lambda x: x != "--" and x, v))
          for k, v in clean_ad_report_data[description_cols]
          .T.to_dict("list")
          .items()
      },
      index=clean_ad_report_data.index,
  )
  clean_ad_report_data = clean_ad_report_data.set_index(
      ["Campaign ID", "Ad Group ID"]
  )[["headlines", "descriptions"]]
  clean_ad_report_data = clean_ad_report_data.loc[
      clean_ad_report_data["headlines"].apply(len) > 0
  ]

  return clean_ad_report_data


def _write_result_into_google_sheet(
    config: Dict[str, str], sheets_client: gspread.client, results: pd.Series
):
  """Writes the generated ad copy results to a Google Sheet.

  Args:
    config: A dictionary containing configuration parameters, including: -
      'RESULTS_URL': The URL of the Google Sheet to write to. -
      'RESULTS_SHEET_NAME': The name of the sheet within the Google Sheet where
      results should be written.
    sheets_client: A gspread.Client object for interacting with Google Sheets.
    results: A pandas Series containing the generated ad copy results.
  """
  results_sheet_name = config["RESULTS_SHEET_NAME"]

  sheet = sheets_client.open_by_url(config["RESULTS_URL"])

  column_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

  values = results.values.tolist()
  values.insert(0, results.columns.values.tolist())

  worksheets = [ws.title for ws in sheet.worksheets()]

  if results_sheet_name not in worksheets:
    sheet.add_worksheet(results_sheet_name, len(values), 100)

  sheet.worksheet(results_sheet_name).update(
      f"A1:{column_letters[len(values[0])-1]}{len(values)+1}", values
  )
