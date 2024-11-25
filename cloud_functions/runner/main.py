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
import logging
import os
import sys
import traceback
from typing import Any
import flask
import functions_framework
import google.auth
from google.cloud import logging as gclogging
import gspread
import nest_asyncio
import pandas as pd

nest_asyncio.apply()

from copycat import copycat
from copycat.data import sheets
from copycat.data import utils as data_utils
from copycat import google_ads

sheet_client = None

DEFAULT_CONFIG = {
    "COMPANY_NAME": "My Company",
    "COPYCAT_ENDPOINT_URL": "",
    "COPYCAT_SERVICE_ACCOUNT": "",
    "TRAINING_ADS_SHEET_NAME": "Training Ads",
    "EMBEDDING_MODEL_NAME": "text-embedding-004",
    "AD_FORMAT": "responsive_search_ad",
    "LANGUAGE": "English",
    "CHAT_MODEL_NAME": "gemini-1.5-pro",
    "MODEL_DIMENSIONALITY": 768,
    "TEMPERATURE": 0.7,
    "TOP_K": 50,
    "TOP_P": 1,
    "BATCH_SIZE": 10,
    "GENERATION_LIMIT": 0,
    "NUM_AD_VERSIONS": 1,
    "USE_CUSTOM_AFFINITY_PREFERENCE": "TRUE",
    "CUSTOM_AFFINITY_PREFERENCE": -0.5,
    "EXEMPLAR_SELECTION_METHOD": "affinity_propagation",
    "MAX_INITIAL_ADS": 2000,
    "MAX_EXEMPLAR_ADS": 200,
    "REPLACE_SPECIAL_VARIABLES_WITH_DEFAULT": "replace",
    "ON_INVALID_AD": "drop",
    "USE_STYLE_GUIDE": "FALSE",
    "STYLE_GUIDE_ADDITIONAL_INSTRUCTIONS": "",
    "STYLE_GUIDE_USE_EXEMPLAR_ADS": "TRUE",
    "STYLE_GUIDE_FILES_URL": "",
    "STYLE_GUIDE": "",
}

FAILED_STATUS_FOR_REPORTING = "ERROR_IN_COPYCAT_RUNNER"
CF_NAME = "copycat_runner"

OPERATIONS = {
    "STYLE_GUIDE_GENERATION": "_style_guide_generation",
    "ADS_GENERATION": "_ads_generation",
}


def _init_log():
  """Initializes the logger."""
  client = gclogging.Client()
  client.setup_logging()
  return logging.getLogger("copycat.on_sheets")


def _send_log(message: str, level: int = logging.INFO) -> None:
  """Sends a log message to the logger.

  Args:
    message: The log message to send.
    level: The level of the log message. Defaults to INFO.
  """

  logger.log(level=level, msg=message)


logger = _init_log()


def _prepare_training_data(
    config: pd.DataFrame, sheet: sheets.GoogleSheet
) -> pd.DataFrame:
  """Prepares training data for the Copycat model.

  Loads the training ads data from the specified sheet, collapses the headlines
  and descriptions
  into lists, renames the 'Keywords' column to 'keywords', selects the relevant
  columns,
  and filters out rows with empty 'headlines'.

  Args:
    config: A Pandas DataFrame containing configuration parameters, including
      the sheet name for training ads.
    sheet: A GoogleSheet object representing the spreadsheet containing the
      data.

  Returns:
    A Pandas DataFrame containing the prepared training data with 'keywords',
    'headlines',
    and 'descriptions' columns.
  """

  training_ads_data = sheet[config["TRAINING_ADS_SHEET_NAME"]]

  prepared_training_ads = data_utils.collapse_headlines_and_descriptions(
      training_ads_data
  ).rename(columns={"Keywords": "keywords"})[
      ["keywords", "headlines", "descriptions"]
  ]

  prepared_training_ads = prepared_training_ads.loc[
      prepared_training_ads["headlines"].apply(len) > 0
  ]

  return prepared_training_ads


def _prepare_new_ads_for_generation(
    sheet: sheets.GoogleSheet,
    n_versions: int,
    fill_gaps: bool,
    copycat_instance: copycat.Copycat,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Prepares new ads for generation.

  This function prepares new ads for generation by loading the data from the
  Google Sheet, constructing the complete data, and filtering out any ads that
  have already been generated.

  Args:
    sheet: The Google Sheet to load the data from.
    n_versions: The number of versions to generate for each ad.
    fill_gaps: Whether to fill gaps in the existing generations.
    copycat_instance: The Copycat instance to use for generation.

  Returns:
    A tuple containing the new generations data and the complete data.
  """
  new_keywords_data = sheet["New Keywords"]
  additional_instructions_data = sheet["Extra Instructions for New Ads"]

  additional_instructions_data.index = (
      additional_instructions_data.index.set_levels(
          additional_instructions_data.index.get_level_values("Version").astype(
              str
          ),
          level="Version",
          verify_integrity=False,
      )
  )

  if "Generated Ads" in sheet:
    existing_generations_data = sheet["Generated Ads"]
    if "Headline 1" not in existing_generations_data.columns:
      existing_generations_data = None
  else:
    existing_generations_data = None

  if existing_generations_data is None:

    complete_data = data_utils.construct_generation_data(
        new_keywords_data=new_keywords_data,
        additional_instructions_data=additional_instructions_data,
        n_versions=n_versions,
        keyword_column="Keyword",
        version_column="Version",
        additional_instructions_column="Extra Instructions",
    )
    complete_data["existing_headlines"] = [[]] * len(complete_data)
    complete_data["existing_descriptions"] = [[]] * len(complete_data)
    new_generations_data = complete_data.copy()
    return new_generations_data, complete_data

  existing_generations_data = data_utils.collapse_headlines_and_descriptions(
      existing_generations_data
  ).rename(
      columns={
          "headlines": "existing_headlines",
          "descriptions": "existing_descriptions",
      }
  )

  existing_generations_data.index = existing_generations_data.index.set_levels(
      existing_generations_data.index.get_level_values("Version").astype(str),
      level="Version",
      verify_integrity=False,
  )

  complete_data = data_utils.construct_generation_data(
      new_keywords_data=new_keywords_data,
      additional_instructions_data=additional_instructions_data,
      existing_generations_data=existing_generations_data,
      n_versions=n_versions,
      existing_headlines_column="existing_headlines",
      existing_descriptions_column="existing_descriptions",
      keyword_column="Keyword",
      version_column="Version",
      additional_instructions_column="Extra Instructions",
  )

  missing_columns = [
      column
      for column in existing_generations_data.columns
      if column not in complete_data.columns
  ]

  if missing_columns:
    complete_data = complete_data.join(
        existing_generations_data[missing_columns],
        how="left",
    )

  generation_not_required = complete_data.index.isin(
      existing_generations_data.index
  )

  if fill_gaps:
    generation_not_required = generation_not_required & complete_data.apply(
        lambda row: copycat_instance.ad_copy_evaluator.is_complete(
            copycat.GoogleAd(
                headlines=row["existing_headlines"],
                descriptions=row["existing_descriptions"],
            )
        ),
        axis=1,
    )
  else:
    generation_not_required = generation_not_required & complete_data.apply(
        lambda row: not copycat_instance.ad_copy_evaluator.is_empty(
            copycat.GoogleAd(
                headlines=row["existing_headlines"],
                descriptions=row["existing_descriptions"],
            )
        ),
        axis=1,
    )

  new_generations_data = complete_data.loc[~generation_not_required].copy()

  return new_generations_data, complete_data


def _style_guide_generation(
    config: dict[str, str],
    config_sheet_name: str,
    sheet: sheets.GoogleSheet,
    copycat_instance: copycat.Copycat,
) -> str:
  """Generates a style guide and writes it to the config sheet.

  This function generates a style guide using the provided Copycat instance and
  configuration parameters. The generated style guide is then stored in the
  config dictionary and written to the specified Google Sheet.

  Args:
    config: A dictionary containing configuration parameters.
    config_sheet_name: The name of the sheet in the Google Sheet containing the
      configuration.
    sheet: A GoogleSheet object representing the spreadsheet.
    copycat_instance: A Copycat object used for generating the style guide.

  Returns:
    The generated style guide string.
  """

  config["STYLE_GUIDE"] = _generate_style_guide(config, copycat_instance)
  _write_config_into_google_sheet(config, sheet, config_sheet_name)

  return config["STYLE_GUIDE"]


def _instantiate_copycat_model(config: pd.DataFrame, sheet: sheets.GoogleSheet):
  """Instantiates a Copycat model.

  Prepares the training data, determines the ad format, and creates a Copycat
  instance using the provided configuration and data from the Google Sheet.

  Args:
    config: A Pandas DataFrame containing the configuration parameters.
    sheet: A GoogleSheet object representing the spreadsheet containing the
      data.

  Returns:
    A Copycat instance.
  """

  training_ads_data = _prepare_training_data(config, sheet)

  max_headlines = google_ads.get_google_ad_format(
      "responsive_search_ad"
  ).max_headlines
  max_descriptions = google_ads.get_google_ad_format(
      "responsive_search_ad"
  ).max_descriptions

  if config["AD_FORMAT"] == "custom":
    ad_format = copycat.google_ads.GoogleAdFormat(
        name="custom",
        max_headlines=max_headlines,
        max_descriptions=max_descriptions,
        min_headlines=1,
        min_descriptions=1,
        max_headline_length=30,
        max_description_length=90,
    )
  else:
    ad_format = copycat.google_ads.get_google_ad_format(config["AD_FORMAT"])

  affinity_preference = (
      float(config["CUSTOM_AFFINITY_PREFERENCE"])
      if config["USE_CUSTOM_AFFINITY_PREFERENCE"].lower() == "true"
      else None
  )

  copycat_instance = copycat.Copycat.create_from_pandas(
      training_data=training_ads_data,
      ad_format=ad_format,
      on_invalid_ad=config["ON_INVALID_AD"],
      embedding_model_name=config["EMBEDDING_MODEL_NAME"],
      embedding_model_dimensionality=float(config["MODEL_DIMENSIONALITY"]),
      embedding_model_batch_size=int(config["BATCH_SIZE"]),
      vectorstore_exemplar_selection_method=config["EXEMPLAR_SELECTION_METHOD"],
      vectorstore_max_initial_ads=int(config["MAX_INITIAL_ADS"]),
      vectorstore_max_exemplar_ads=int(config["MAX_EXEMPLAR_ADS"]),
      vectorstore_affinity_preference=affinity_preference,
      replace_special_variables_with_default=config[
          "REPLACE_SPECIAL_VARIABLES_WITH_DEFAULT"
      ]
      == "replace",
  )

  return copycat_instance


def _ads_generation(
    config: dict[str, str],
    config_sheet_name: str,
    sheet: sheets.GoogleSheet,
    copycat_instance: copycat.Copycat,
):
  """Generates new ads using the Copycat model and writes them to the Google Sheet.

  This function orchestrates the ad generation process. It retrieves the ad
  format,
  generates or retrieves a style guide, prepares the data for generation,
  generates new ads in batches using the Copycat model, and writes the generated
  ads to the "Generated Ads" sheet in the Google Sheet.

  Args:
    config: A dictionary containing the configuration parameters.
    config_sheet_name: The name of the sheet containing configuration
      parameters.
    sheet: The GoogleSheet object representing the spreadsheet.
    copycat_instance: The instantiated Copycat model.
  """
  max_headlines = google_ads.get_google_ad_format(
      "responsive_search_ad"
  ).max_headlines
  max_descriptions = google_ads.get_google_ad_format(
      "responsive_search_ad"
  ).max_descriptions

  style_guide = ""
  if config["USE_STYLE_GUIDE"].lower() == "true":
    style_guide = config["STYLE_GUIDE"] if "STYLE_GUIDE" in config else ""

  if not (style_guide and len(style_guide) > 1):
    style_guide = _style_guide_generation(
        config, config_sheet_name, sheet, copycat_instance
    )

  generation_data, complete_data = _prepare_new_ads_for_generation(
      sheet,
      int(config["NUM_AD_VERSIONS"]),
      fill_gaps=True,
      copycat_instance=copycat_instance
  )

  updated_complete_data = data_utils.explode_headlines_and_descriptions(
      complete_data.copy().rename(
          columns={
              "existing_headlines": "headlines",
              "existing_descriptions": "descriptions",
          }
      ),
      max_headlines=max_headlines,
      max_descriptions=max_descriptions,
  )
  _send_log("Loaded generation and complete data")

  if len(generation_data) == 0:
    _send_log("No ads to generate")
    return

  generation_params = dict(
      system_instruction_kwargs=dict(
          company_name=config["COMPANY_NAME"],
          language=config["LANGUAGE"],
      ),
      num_in_context_examples=int(config["NUM_IN_CONTEXT_EXAMPLES"]),
      model_name=config["CHAT_MODEL_NAME"],
      temperature=float(config["TEMPERATURE"]),
      top_k=float(config["TOP_K"]),
      top_p=float(config["TOP_P"]),
      allow_memorised_headlines=config["ALLOW_MEMORISED_HEADLINES"].lower()
      == "yes",
      allow_memorised_descriptions=config[
          "ALLOW_MEMORISED_DESCRIPTIONS"
      ].lower()
      == "yes",
      safety_settings=copycat.ALL_SAFETY_SETTINGS_ONLY_HIGH,
      style_guide=style_guide,
  )
  limit = int(config["GENERATION_LIMIT"])
  if limit == 0:
    limit = None
  data_iterator = data_utils.iterate_over_batches(
      generation_data,
      batch_size=int(config["BATCH_SIZE"]),
      limit_rows=limit,
  )
  for batch_number, generation_batch in enumerate(data_iterator):
    _send_log(f"Generating batch {batch_number+1}")
    generation_batch["generated_ad_object"] = (
        copycat_instance.generate_new_ad_copy_for_dataframe(
            data=generation_batch,
            keywords_specific_instructions_column="additional_instructions",
            **generation_params,
        )
    )
    generation_batch = (
        generation_batch.pipe(data_utils.explode_generated_ad_object)
        .pipe(
            data_utils.explode_headlines_and_descriptions,
            max_headlines=max_headlines,
            max_descriptions=max_descriptions,
        )
        .drop(
            columns=[
                "generated_ad_object",
                "existing_headlines",
                "existing_descriptions",
            ],
            errors="ignore",
        )
    )

    isin_batch = updated_complete_data.index.isin(generation_batch.index)

    updated_complete_data = updated_complete_data.loc[~isin_batch]
    updated_complete_data = pd.concat([updated_complete_data, generation_batch])
    updated_complete_data = updated_complete_data.fillna("").loc[
        complete_data.index
    ]

    column_order = ["keywords", "additional_instructions"]
    column_order.extend(
        col
        for col in updated_complete_data.columns
        if col.startswith("Headline ")
    )
    column_order.extend(
        col
        for col in updated_complete_data.columns
        if col.startswith("Description ")
    )
    column_order.extend(
        col for col in updated_complete_data.columns if col not in column_order
    )

    sheet["Generated Ads"] = updated_complete_data[column_order]

  _send_log("Generation Complete")


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
  required_elem = [
      "PROJECT_ID",
      "REGION",
      "SERVICE_ACCOUNT",
      "DEPLOYMENT_NAME",
  ]
  if not all(elem in os.environ for elem in required_elem):
    _send_log(
        f"{FAILED_STATUS_FOR_REPORTING}: Cannot proceed, there are missing"
        " input values please make sure you set all the environment variables"
        " correctly."
    )

    sys.exit(1)

  _send_log(request)
  request_json = request.get_json(silent=True)
  _send_log(request_json)

  try:

    operation = request_json["operation"]
    print(f"Operation: {operation}.")
    function_name = (
        OPERATIONS[operation] if operation in OPERATIONS else OPERATIONS[0]
    )
    print(f"Function name: {function_name}.")
    worksheet_url = request_json["worksheet_url"]
    config_sheet_name = request_json["config_sheet_name"]
    _init_google_sheet_client()
    sheet = sheets.GoogleSheet.load(worksheet_url)

    config = _read_config_from_google_sheet(
        DEFAULT_CONFIG,
        sheets.get_gspread_client(),
        worksheet_url,
        config_sheet_name,
    )

    copycat_instance = _instantiate_copycat_model(config, sheet)

    call_function = globals()[function_name]
    call_function(config, config_sheet_name, sheet, copycat_instance)

    return ("Results generated", 200)
  except Exception as e:

    print(traceback.format_exc())
    _send_log(str(e))
    logging_payload = {
        "worksheet_url": worksheet_url if worksheet_url else None,
        "config_sheet_name": config_sheet_name if config_sheet_name else None,
        "status": FAILED_STATUS_FOR_REPORTING,
        "message": (
            str(e)
            if len(str(e)) > 1
            else "Check you shared the spreadsheet with the service account"
        ),
        "success": False,
    }

    _send_log(f"{FAILED_STATUS_FOR_REPORTING}: {json.dumps(logging_payload)}")

    return (json.dumps(logging_payload), 500)


def _generate_style_guide(
    config: dict[str, Any], copycat_instance: copycat.Copycat
) -> str:
  """Generates a style guide using the Copycat model.

  Leverages the Copycat instance's `generate_style_guide` method to create a
  style
  guide based on the provided configuration parameters.

  Args:
    config: A dictionary containing configuration parameters, including: -
      "COMPANY_NAME": The name of the company. -
      "STYLE_GUIDE_ADDITIONAL_INSTRUCTIONS": Additional instructions for style
      guide generation. - "CHAT_MODEL_NAME": The name of the language model to
      use. - "TEMPERATURE": The temperature setting for the language model. -
      "TOP_K": The top_k setting for the language model. - "TOP_P": The top_p
      setting for the language model. - "STYLE_GUIDE_USE_EXEMPLAR_ADS": Whether
      to use exemplar ads. - "STYLE_GUIDE_FILES_URL":  The URI of files to use
      for the style guide.
    copycat_instance: An instantiated Copycat model.

  Returns:
    A string containing the generated style guide.
  """

  style_guide = copycat_instance.generate_style_guide(
      company_name=config["COMPANY_NAME"],
      additional_style_instructions=config[
          "STYLE_GUIDE_ADDITIONAL_INSTRUCTIONS"
      ],
      model_name=config["CHAT_MODEL_NAME"],
      safety_settings=copycat.ALL_SAFETY_SETTINGS_ONLY_HIGH,
      temperature=float(config["TEMPERATURE"]),
      top_k=float(config["TOP_K"]),
      top_p=float(config["TOP_P"]),
      use_exemplar_ads=config["STYLE_GUIDE_USE_EXEMPLAR_ADS"],
      files_uri=config["STYLE_GUIDE_FILES_URL"],
  )

  return style_guide


def _init_google_sheet_client():
  """Initializes and authenticates a gspread client.

  This function sets up the necessary credentials and authorization
  to interact with Google Sheets using the gspread library.
  """
  scopes = [
      "https://www.googleapis.com/auth/spreadsheets",
      "https://www.googleapis.com/auth/drive",
      "https://spreadsheets.google.com/feeds",
  ]

  credentials, _ = google.auth.default(scopes=scopes)
  sheets.set_google_auth_credentials(credentials)


def _read_config_from_google_sheet(
    default_config: dict[str, str],
    sheets_client: gspread.Client,
    worksheet_url: str,
    config_sheet_name: str,
) -> dict[str, str]:
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


def _write_config_into_google_sheet(
    config: dict[str, str],
    sheet: sheets.GoogleSheet,
    config_sheet_name: str
):
  """Writes the configuration parameters to a Google Sheet.

  Args:
    config: A dictionary containing the configuration parameters.
    sheet: GoogleSheet object.
    config_sheet_name: The name of the sheet in the Google Sheet containing the
      configuration.
  """
  worksheet_url = sheet.url
  client = sheets.get_gspread_client()
  worksheet = client.open_by_url(worksheet_url)
  config_sheet = worksheet.worksheet(config_sheet_name)

  cell = config_sheet.find("STYLE_GUIDE")
  if cell:
    config_sheet.update_cell(cell.row, cell.col + 1, config["STYLE_GUIDE"])
