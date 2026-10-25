"""
File: data_clean.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: data cleaning
Relative Path: /data_clean.py
File Created: Monday, 4th April 2022 10:18:17 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Thursday, 14th April 2022 9:17:03 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2022, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import re
import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

# read arguments
parser = argparse.ArgumentParser(description="set arguments")
parser.add_argument("--source_file", type=str, help="source file path, json or csv")
parser.add_argument("--output_path", default="", type=str, help="output file path")
parser.add_argument(
    "--output_file", type=str, help="output file default name, csv only"
)
parser.add_argument(
    "--format_table",
    default="flatten_special_case.xlsx",
    type=str,
    help="helper excel file path",
)
args = parser.parse_args()

WEEKDAY = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def str2list(item):

    try:
        return ast.literal_eval(item)
    #         return json.loads(item)
    except:
        return item


def string2int(item):

    try:
        return int(item)
    except:
        return item


def get_hour(item):

    try:
        return int(item / 100)
    except:
        return item


def check_str(item, sub=""):

    if isinstance(item, str):
        if sub in item:
            return True
        else:
            return False
    else:
        return False


# check whether language in the item
# from a list of lanugage to [0, 1] list
def apply_top_k(item, top_k_language):

    if isinstance(item, list):
        result = []
        for _language in top_k_language:
            result.append(int(_language in item))
        result.append(int(len(list(set(item) - set(top_k_language))) > 0))
    else:
        return [0 for _ in range(len(top_k_language) + 1)]

    return result


# format language (split by comma, find unique, remove () part)
def format_language(item):

    result = []

    if isinstance(item, list):
        for _item in item:
            result.append(_item.split(","))
    elif isinstance(item, str):
        result.append(item.split(","))
    else:
        return item

    # flatten data
    result = list(set(np.concatenate(result).flat))
    # remove () part and find unique value
    result = [re.sub("[\(\[].*?[\)\]]", "", _item.replace(" ", "")) for _item in result]

    return result


# find standard cleaning methods
def get_cleaning(file_path, sheet_name="output"):

    output = pd.read_excel(file_path, sheet_name=sheet_name)
    output = output.loc[:, output.columns.isin(["name", "cleaning_method"])]

    return output


# find standard cleaning methods
def get_mapping(file_path, sheet_name="special_mapping"):

    output = pd.read_excel(file_path, sheet_name=sheet_name)

    return output


def get_range(file_path, sheet_name="special_range"):

    output = pd.read_excel(file_path, sheet_name=sheet_name)
    output = output.loc[:, ~output.columns.isin(["P.S."])]

    return output


def cleaning_mail_deliverable_status(data):

    data = data.apply(lambda x: str2list(x))

    def unify_type(item):

        if isinstance(item, list):
            # lower all text in case upper/lower case issue
            if "deliverable" in [_item.lower() for _item in item]:
                return "Deliverable"
            else:
                return "Questionable"
        elif isinstance(item, str):
            # lower all text in case upper/lower case issue
            if item.lower() == "deliverable":
                return "Deliverable"
            else:
                return "Questionable"

    return data.apply(lambda x: unify_type(x))


def cleaning_drop(data, cleaning_format):

    drop_cols = cleaning_format.loc[
        cleaning_format["cleaning_method"] == "drop", "name"
    ]

    return data.drop(columns=drop_cols)


def cleaning_mapping(data):

    # read grouping/mapping rules for this column
    # dataframe to dict
    mapping_rule = mapping_format[
        [data.name + "_original", data.name + "_output"]
    ].dropna(
        how="all"
    )  # due to rule length inconsistency, need to drop nan rows

    mapping_rule = dict(
        zip(mapping_rule[data.name + "_original"], mapping_rule[data.name + "_output"])
    )

    # apply actual mapping rule
    return data.map(mapping_rule, na_action="ignore")


def cleaning_language_spoken(data, top_k=5):

    # get top-k frequency
    # format language spoken
    data = data.apply(lambda x: str2list(x))  # get to list if possible
    data = data.apply(lambda x: format_language(x))

    unique_language = {}
    for item in data:
        if isinstance(item, list):
            for _item in item:
                if _item not in unique_language.keys():
                    unique_language[_item] = 1
                else:
                    unique_language[_item] += 1

    # sort by frequency
    unique_language = {
        k: v for k, v in sorted(unique_language.items(), key=lambda item: item[1])
    }
    # get top-5 language spoken
    top_k_language = list(unique_language.keys())[-top_k:]

    # list of string language to list of [0, 1]
    data = data.apply(lambda x: apply_top_k(x, top_k_language))

    # get column names (top_k + Other)
    columns = [data.name + "_" + _language for _language in top_k_language]
    columns.append(data.name + "_Other")

    return pd.DataFrame(data.tolist(), columns=columns)


def clean_main(data):

    # drop columns
    drop_cols = cleaning_format.loc[
        cleaning_format["cleaning_method"] == "drop", "name"
    ]
    data.drop(columns=drop_cols, inplace=True)

    # professionals_range
    # only convert to string here
    # will be mapped below
    data["FIRMO.professionals_range"] = data["FIRMO.professionals_range"].str.strip()

    # FIRMO.employee_range special treatment
    data["FIRMO.employee_range"] = data["FIRMO.employee_range"].str.replace(" TO ", "-")
    data["FIRMO.employee_range"] = data["FIRMO.employee_range"].str.replace("Oct", "10")
    data["FIRMO.employee_range"] = data["FIRMO.employee_range"].str.replace(
        " employees", ""
    )
    data["FIRMO.employee_range"] = data["FIRMO.employee_range"].str.replace(
        " Employees", ""
    )
    data["FIRMO.employee_range"] = data["FIRMO.employee_range"].str.replace(
        " employee", ""
    )

    # special mapping/grouping
    mapping_cols = cleaning_format.loc[
        cleaning_format["cleaning_method"].apply(lambda x: check_str(x, "mapping")),
        "name",
    ]
    for column in mapping_cols:
        data[column] = cleaning_mapping(data[column])

    # special cases
    # ADDRESSES.mail_deliverable_status_type
    data["ADDRESSES.mail_deliverable_status_type"] = cleaning_mail_deliverable_status(
        data["ADDRESSES.mail_deliverable_status_type"]
    )

    # # operating_year
    # data["BUSINESS.operating_year"] = data["YEAR"] - data["BUSINESS.founded_year"]
    
    # delete founded_year
    # data.drop(columns=["BUSINESS.founded_year"], inplace=True)

    # convert language spoken
    language_data = cleaning_language_spoken(data["BUSINESS.languages_spoken"])
    data.drop(columns=["BUSINESS.languages_spoken"], inplace=True)
    data = pd.concat([data, language_data], axis=1)

    # CARPE_INDEX
    # use newest scoring
    # get sources from CARPE_INDEXES
    all_carpe_indexes = data.columns[
        data.columns.str.contains("CARPE_INDEXES")
    ]  # get all CARPE_INDEXES
    all_carpe_indexes_source = [
        item.split(".")[1] for item in all_carpe_indexes
    ]  # split to get source (source is the second item)
    all_carpe_indexes_source = list(set(all_carpe_indexes_source))  # get unique sources
    all_carpe_indexes_source.sort()  # sort by alphabet

    # basic structure, if already get newest scoring (use old version to fillna)
    #                  if not, use the current scoring as newest scoring
    for source in all_carpe_indexes_source:
        # check if version 2.0 is available
        if "CARPE_INDEXES." + source + ".score_version_2.0" in data.columns:
            # if already get newest scoring, use current version to fillna
            if "CARPE_INDEXES." + source + ".score_version_newest" in data.columns:
                data["CARPE_INDEXES." + source + ".score_version_newest"] = data[
                    "CARPE_INDEXES." + source + ".score_version_newest"
                ].fillna(data["CARPE_INDEXES." + source + ".score_version_2.0"])
            # if not, use current version as newest scoring
            else:
                data["CARPE_INDEXES." + source + ".score_version_newest"] = data[
                    "CARPE_INDEXES." + source + ".score_version_2.0"
                ]
        # check if version 1.1 is available
        if "CARPE_INDEXES." + source + ".score_version_1.1" in data.columns:
            # check if newest version is available
            if "CARPE_INDEXES." + source + ".score_version_newest" in data.columns:
                data["CARPE_INDEXES." + source + ".score_version_newest"] = data[
                    "CARPE_INDEXES." + source + ".score_version_newest"
                ].fillna(data["CARPE_INDEXES." + source + ".score_version_1.1"])
            else:
                data["CARPE_INDEXES." + source + ".score_version_newest"] = data[
                    "CARPE_INDEXES." + source + ".score_version_1.1"
                ]
        # check if version 1.0 is available
        if "CARPE_INDEXES." + source + ".score_version_1.0" in data.columns:
            # check if newest version is available
            if "CARPE_INDEXES." + source + ".score_version_newest" in data.columns:
                data["CARPE_INDEXES." + source + ".score_version_newest"] = data[
                    "CARPE_INDEXES." + source + ".score_version_newest"
                ].fillna(data["CARPE_INDEXES." + source + ".score_version_1.0"])
            else:
                data["CARPE_INDEXES." + source + ".score_version_newest"] = data[
                    "CARPE_INDEXES." + source + ".score_version_1.0"
                ]

    # remove old scoring systems
    data.drop(
        columns=data.columns[
            data.columns.str.match("^CARPE_INDEXES.+\.score_version_1.0$")
        ],
        inplace=True,
    )
    data.drop(
        columns=data.columns[
            data.columns.str.match("^CARPE_INDEXES.+\.score_version_1.1$")
        ],
        inplace=True,
    )
    data.drop(
        columns=data.columns[
            data.columns.str.match("^CARPE_INDEXES.+\.score_version_2.0$")
        ],
        inplace=True,
    )

    # corp_revenue
    data["FIRMO.corp_revenue_useNA"] = data["FIRMO.corp_revenue"]
    data["FIRMO.corp_revenue_use0"] = data["FIRMO.corp_revenue"].fillna(0)
    data.drop(columns=["FIRMO.corp_revenue"], inplace=True)

    # cross reference
    # cross reference is_temporarily_closed with is_closed
    data["BUSINESS.is_closed"] = data["BUSINESS.is_closed"].fillna(
        data["BUSINESS.is_temporarily_closed"]
    )
    data.drop(columns="BUSINESS.is_temporarily_closed", inplace=True)

    # FIRMO.employee_range reference on FIRMO.employee_on_site
    # get LICENSING_count range mapping
    employee_mapping = range_format.loc[
        range_format["name"] == "FIRMO.employee_on_site"
    ].dropna(how="all", axis=1)

    employee_header = employee_mapping.iloc[0, 1:]  # grab the first row for the header
    employee_mapping = employee_mapping.iloc[
        1:, 1:
    ]  # take the data less the header row
    employee_mapping.columns = employee_header  # set the header row as the df header
    # unique 0
    for left_range, right_range, mapping in zip(
        employee_mapping["left_range"],
        employee_mapping["right_range"],
        employee_mapping["mapping"],
    ):
        if str(left_range) == "nan":
            exec(
                "data.loc[data['FIRMO.employee_on_site'] "
                + right_range
                + " , 'FIRMO.employee_range'] = '"
                + mapping
                + "'"
            )
        elif str(right_range) == "nan":
            exec(
                "data.loc[data['FIRMO.employee_on_site'] "
                + left_range
                + " , 'FIRMO.employee_range'] = '"
                + mapping
                + "'"
            )
        else:
            exec(
                "data.loc[((data['FIRMO.employee_on_site']"
                + left_range
                + ") \
                    & (data['FIRMO.employee_on_site']"
                + right_range
                + ")) \
                    , 'FIRMO.employee_range' ] = '"
                + mapping
                + "'"
            )

    # employee_range cross reference with employee_at_site
    data.drop(columns=["FIRMO.employee_on_site"], inplace=True)

    # FIRMO.revenue_range reference on FIRMO.revenue_at_site
    # convert to integer if possible
    data["FIRMO.revenue_at_site"] = data["FIRMO.revenue_at_site"].apply(
        lambda x: string2int(x)
    )
    # unify string format
    data.loc[
        (data["FIRMO.revenue_at_site"] == "under1Mil"), "FIRMO.revenue_at_site"
    ] = "under1Mil"
    data.loc[
        (data["FIRMO.revenue_at_site"] == "from1_4p9Mil"), "FIRMO.revenue_at_site"
    ] = "1-4.9Mil"
    data.loc[
        (data["FIRMO.revenue_at_site"] == "from5_9p9Mil"), "FIRMO.revenue_at_site"
    ] = "5-9.9Mil"
    data.loc[
        (data["FIRMO.revenue_at_site"] == "from10_99p9Mil"), "FIRMO.revenue_at_site"
    ] = "10-99.9Mil"
    data.loc[
        (data["FIRMO.revenue_at_site"] == "from100_249p9Mil"), "FIRMO.revenue_at_site"
    ] = "100-249.9Mil"
    data.loc[
        (data["FIRMO.revenue_at_site"] == "from250Milover"), "FIRMO.revenue_at_site"
    ] = "250Milover"

    # try:
    #     data["FIRMO.revenue_at_site"] = data["FIRMO.revenue_at_site"].astype(float)
    # except:
    #     raise ValueError(
    #         "Get unseen observations in FIRMO.revenue_at_site, add to special_mapping!"
    #     )

    # revenue_range cross reference with revenue_at_site
    data["FIRMO.revenue_range"] = data["FIRMO.revenue_range"].fillna(
        data["FIRMO.revenue_at_site"]
    )
    data.drop(columns=["FIRMO.revenue_at_site"], inplace=True)

    # data = pd.concat([data, pd.get_dummies(data[["FIRMO.revenue_range"]])], axis=1)
    # data.drop(columns="FIRMO.revenue_range", inplace=True)

    # grouping LICENSING_count
    # get LICENSING_count range mapping
    license_mapping = range_format.loc[
        range_format["name"] == "LICENSING_count"
    ].dropna(how="all", axis=1)

    license_header = license_mapping.iloc[0, 1:]  # grab the first row for the header
    license_mapping = license_mapping.iloc[1:, 1:]  # take the data less the header row
    license_mapping.columns = license_header  # set the header row as the df header

    # use rules to map LICENSING_count to LICENSING_count_group
    for _rule, _mapping in zip(license_mapping["rule"], license_mapping["mapping"]):
        exec(
            "data.loc[data.LICENSING_count"
            + _rule
            + ", 'LICENSING_count_group'] = '"
            + _mapping
            + "'"
        )

    data = pd.concat([data, pd.get_dummies(data[["LICENSING_count_group"]])], axis=1)
    data.drop(columns=["LICENSING_count", "LICENSING_count_group"], inplace=True)

    ## REVIEW_SCORE

    ## Average Score
    # only booking, expedia, facebook, foursquare have both reviews_count and rating_counts
    all_review_score = data.columns[data.columns.str.contains("REVIEW_SCORE")]
    all_review_score_sources = [
        item.split(".")[1] for item in all_review_score
    ]  # split to get source (source is the second item)
    all_review_score_sources = list(set(all_review_score_sources))  # remove duplicates
    all_review_score_sources.sort()  # sort by alphabetical order

    # loop through all review_score sources
    for source in all_review_score_sources:
        # reviews_count = max(reviews_count, rating_counts) if both exist, and use the one if only one exists
        if (
            "REVIEW_SCORE." + source + ".reviews_count" in data.columns
            and "REVIEW_SCORE." + source + ".rating_counts" in data.columns
        ):
            data["REVIEW_SCORE." + source + ".reviews_count"] = data[
                [
                    "REVIEW_SCORE." + source + ".rating_counts",
                    "REVIEW_SCORE." + source + ".reviews_count",
                ]
            ].max(axis=1)
            data.drop(
                columns=["REVIEW_SCORE." + source + ".rating_counts"], inplace=True
            )
        elif (
            "REVIEW_SCORE." + source + ".reviews_count" in data.columns
            and "REVIEW_SCORE." + source + ".rating_counts" not in data.columns
        ):
            pass
        elif (
            "REVIEW_SCORE." + source + ".reviews_count" not in data.columns
            and "REVIEW_SCORE." + source + ".rating_counts" in data.columns
        ):
            data["REVIEW_SCORE." + source + ".reviews_count"] = data[
                "REVIEW_SCORE." + source + ".rating_counts"
            ]
            data.drop(
                columns=["REVIEW_SCORE." + source + ".rating_counts"], inplace=True
            )

        # scaling score that are not on 0-5 scale
        data["REVIEW_SCORE." + source + ".reviews_score"] = minmax_scale(
            data["REVIEW_SCORE." + source + ".reviews_score"], feature_range=(0, 5)
        )

    # get list of all reviews_count
    reviews_count = np.sort(
        data.columns[data.columns.str.match("^REVIEW_SCORE.+\.reviews_count$")]
    )
    # get list of all reviews_score
    reviews_score = np.sort(
        data.columns[data.columns.str.match("^REVIEW_SCORE.+reviews_score$")]
    )

    # loop though all reviews_count and reviews_score
    # make sure reviews_count and reviews_score are corresponded
    for source in all_review_score_sources:
        if (
            "REVIEW_SCORE." + source + ".reviews_count" in reviews_count
            and "REVIEW_SCORE." + source + ".reviews_score" not in reviews_score
        ):
            reviews_count = np.delete(
                reviews_count,
                np.where(
                    (reviews_count == "REVIEW_SCORE." + source + ".reviews_count")
                ),
            )
        elif (
            "REVIEW_SCORE." + source + ".reviews_count" not in reviews_count
            and "REVIEW_SCORE." + source + ".reviews_score" in reviews_score
        ):
            reviews_score = np.delete(
                reviews_score,
                np.where(
                    (reviews_score == "REVIEW_SCORE." + source + ".reviews_score")
                ),
            )

    # calculating average score
    score = data[reviews_score]
    count = data[reviews_count]

    data["REVIEW_SCORE.average_score"] = np.nansum(
        np.multiply(score, count), axis=1
    ) / np.nansum(np.multiply(score == score, count), axis=1)

    # drop all reviews_count and reviews_score
    data.drop(columns=reviews_score, inplace=True)
    data.drop(columns=reviews_count, inplace=True)

    #########################################################################################################################
    ## Average Rating

    all_rating = data.columns[
        data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.+stars$")
    ]
    rating_sources = (
        pd.Series(
            data.columns[
                data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.+stars$")
            ]
        )
        .str.extract(r"REVIEW_SCORE\.(\w+)\.star_percentage", expand=False)
        .unique()
    )

    # replace inf with 1
    data[all_rating] = data[all_rating].replace(np.inf, 1)

    # for each source
    for source in rating_sources:
        ratings_from_source = all_rating[
            all_rating.str.match(f"^REVIEW_SCORE.{source}.star_percentage.+stars$")
        ]
        # calculate factor
        data["rating_factor"] = np.nansum(data[ratings_from_source], axis=1)
        data["rating_factor"].replace(0, np.nan, inplace=True)
        data["rating_factor"] = 100 / data["rating_factor"]
        for col in ratings_from_source:
            data[col] = data[col] * data["rating_factor"]

    data.drop(columns=["rating_factor"], inplace=True)

    # calculate average rating
    data["REVIEW_SCORE.average_five_stars_rating"] = data[
        data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.five_stars$")
        ]
    ].mean(axis=1)
    data["REVIEW_SCORE.average_four_stars_rating"] = data[
        data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.four_stars$")
        ]
    ].mean(axis=1)
    data["REVIEW_SCORE.average_three_stars_rating"] = data[
        data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.three_stars$")
        ]
    ].mean(axis=1)
    data["REVIEW_SCORE.average_two_stars_rating"] = data[
        data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.two_stars$")
        ]
    ].mean(axis=1)
    data["REVIEW_SCORE.average_one_stars_rating"] = data[
        data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.one_star$")
        ]
    ].mean(axis=1)
    data.dropna(axis=1, how="all", inplace=True)

    # drop all sources star ratings
    data.drop(
        columns=data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.+stars$")
        ],
        inplace=True,
    )
    data.drop(
        columns=data.columns[
            data.columns.str.match("^REVIEW_SCORE.+\.star_percentage.+star$")
        ],
        inplace=True,
    )

    # OHE (keep missing values)
    ohe_cols = cleaning_format.loc[
        cleaning_format["cleaning_method"] == "one-hot", "name"
    ]

    # data["YEAR"] = data["YEAR"].astype(str)
    data = pd.concat([data, pd.get_dummies(data[ohe_cols], dummy_na=True)], axis=1)
    data.drop(columns=ohe_cols, inplace=True)

    # if not data["YEAR_nan"].isnull().any():
    #     data.drop(columns=["YEAR_nan"], inplace=True)

    # OHE (drop missing values)
    ohe_drop_cols = list(
        cleaning_format.loc[
            cleaning_format["cleaning_method"].apply(
                lambda x: check_str(x, "one-hot-drop")
            ),
            "name",
        ]
    )

    # special case for OHE drop missing
    data = pd.concat([data, pd.get_dummies(data[ohe_drop_cols])], axis=1)
    data.drop(columns=ohe_drop_cols, inplace=True)

    # drop OPEN_HOURS.*.status_open
    data.drop(
        columns=data.columns[data.columns.str.match("^OPEN_HOURS.+\.status_open")],
        inplace=True,
    )

    # get open hours special range
    open_hour_mapping = range_format.loc[range_format["name"] == "OPEN_HOURS"].dropna(
        how="all", axis=1
    )

    open_hour_header = open_hour_mapping.iloc[
        0, 1:
    ]  # grab the first row for the header
    open_hour_mapping = open_hour_mapping.iloc[
        1:, 1:
    ]  # take the data less the header row
    open_hour_mapping.columns = open_hour_header  # set the header row as the df header

    # use open_hour/close_hour cross reference status_closed
    for day in WEEKDAY:
        # cross reference the status_closed with open_hour/close_hour
        data.loc[
            data["OPEN_HOURS." + day + ".status_closed"].isnull()
            & data["OPEN_HOURS." + day + ".open_hour"]
            == "closed",
            "OPEN_HOURS." + day + ".status_closed",
        ] = "closed"
        data.loc[
            data["OPEN_HOURS." + day + ".status_closed"].isnull()
            & data["OPEN_HOURS." + day + ".closed_hour"]
            == "closed",
            "OPEN_HOURS." + day + ".status_closed",
        ] = "closed"

        # drop "closed" in open_hour and closed_hour
        data.loc[
            data["OPEN_HOURS." + day + ".open_hour"] == "closed",
            "OPEN_HOURS." + day + ".open_hour",
        ] = np.nan
        data.loc[
            data["OPEN_HOURS." + day + ".closed_hour"] == "closed",
            "OPEN_HOURS." + day + ".closed_hour",
        ] = np.nan

        # force open hour/close hour to float
        try:
            data["OPEN_HOURS." + day + ".open_hour"] = data[
                "OPEN_HOURS." + day + ".open_hour"
            ].astype(float)
        except:
            raise ValueError(
                "Find unexpected string in column {}.".format(
                    "OPEN_HOURS." + day + ".open_hour"
                )
            )

        try:
            data["OPEN_HOURS." + day + ".closed_hour"] = data[
                "OPEN_HOURS." + day + ".closed_hour"
            ].astype(float)
        except:
            raise ValueError(
                "Find unexpected string in column {}.".format(
                    "OPEN_HOURS." + day + ".closed_hour"
                )
            )

        # get all special ranges
        (
            weekday,
            early_open_start,
            early_open_end,
            early_close_start,
            early_close_end,
            late_close_start,
            late_close_end,
        ) = open_hour_mapping.loc[open_hour_mapping["weekday"] == day].values[0]
        # get early_open
        # if left range number > right range number, use | (or)
        # else, use & (and)
        early_open_left_num = int(early_open_start.strip(">").strip("<").strip("="))
        early_open_right_num = int(early_open_end.strip(">").strip("<").strip("="))
        if early_open_left_num <= early_open_right_num:
            exec(
                "data.loc[(data['OPEN_HOURS."
                + day
                + ".open_hour'].apply(lambda x: get_hour(x))"
                + early_open_start
                + ") & (data['OPEN_HOURS."
                + day
                + ".open_hour'].apply(lambda x: get_hour(x)) "
                + early_open_end
                + "), 'OPEN_HOURS."
                + day
                + ".early_open'] = 1"
            )
        else:
            exec(
                "data.loc[(data['OPEN_HOURS."
                + day
                + ".open_hour'].apply(lambda x: get_hour(x))"
                + early_open_start
                + ") | (data['OPEN_HOURS."
                + day
                + ".open_hour'].apply(lambda x: get_hour(x)) "
                + early_open_end
                + "), 'OPEN_HOURS."
                + day
                + ".early_open'] = 1"
            )
        early_close_left_num = int(early_close_start.strip(">").strip("<").strip("="))
        early_close_right_num = int(early_close_end.strip(">").strip("<").strip("="))
        if early_close_left_num <= early_close_right_num:
            exec(
                "data.loc[(data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x))"
                + early_close_start
                + ") & (data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x)) "
                + early_close_end
                + "), 'OPEN_HOURS."
                + day
                + ".early_close'] = 1"
            )
        else:
            exec(
                "data.loc[(data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x))"
                + early_close_start
                + ") | (data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x)) "
                + early_close_end
                + "), 'OPEN_HOURS."
                + day
                + ".early_close'] = 1"
            )
        late_close_left_num = int(late_close_start.strip(">").strip("<").strip("="))
        late_close_right_num = int(late_close_end.strip(">").strip("<").strip("="))
        if late_close_left_num <= late_close_right_num:
            exec(
                "data.loc[(data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x))"
                + late_close_start
                + ") & (data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x)) "
                + late_close_end
                + "), 'OPEN_HOURS."
                + day
                + ".late_close'] = 1"
            )
        else:
            exec(
                "data.loc[(data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x))"
                + late_close_start
                + ") | (data['OPEN_HOURS."
                + day
                + ".closed_hour'].apply(lambda x: get_hour(x)) "
                + late_close_end
                + "), 'OPEN_HOURS."
                + day
                + ".late_close'] = 1"
            )

        # impute missing as 0
        data[
            [
                "OPEN_HOURS." + day + ".early_open",
                "OPEN_HOURS." + day + ".early_close",
                "OPEN_HOURS." + day + ".late_close",
            ]
        ] = data[
            [
                "OPEN_HOURS." + day + ".early_open",
                "OPEN_HOURS." + day + ".early_close",
                "OPEN_HOURS." + day + ".late_close",
            ]
        ].fillna(0)

        data.drop(
            columns=[
                "OPEN_HOURS." + day + ".open_hour",
                "OPEN_HOURS." + day + ".closed_hour",
            ],
            inplace=True,
        )

    # impute with 0
    impute_0_cols = cleaning_format.loc[
        cleaning_format["cleaning_method"] == "impute_0", "name"
    ]
    data[impute_0_cols] = data[impute_0_cols].fillna(0)

    return data


if __name__ == "__main__":

    # get args
    SOURCE_FILE = args.source_file
    OUTPUT_PATH = args.output_path
    OUTPUT_FILE = args.output_file
    FORMAT_TABLE = args.format_table

    data = pd.read_csv(SOURCE_FILE, low_memory=False)

    # get cleaning methods
    cleaning_format = get_cleaning(
        FORMAT_TABLE, sheet_name="output"
    )  # dataframe: columns ["names", "cleaning_method"]

    # get mapping rules
    mapping_format = get_mapping(
        FORMAT_TABLE, sheet_name="special_mapping"
    )  # dataframe: columns ["names", "cleaning_method"]

    # get special range
    range_format = get_range(FORMAT_TABLE, sheet_name="special_range")

    print("Cleaning start...")
    data = clean_main(data)

    # sort columns
    sort_columns = list(data.columns)
    sort_columns.sort()

    # export data
    data[sort_columns].to_csv(os.path.join(OUTPUT_PATH, OUTPUT_FILE), index=False)
