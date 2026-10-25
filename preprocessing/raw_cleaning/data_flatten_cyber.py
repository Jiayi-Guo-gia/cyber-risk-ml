"""
File: data_flatten.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: Downloads
Relative Path: \data_flatten_cyber.py
File Created: Tuesday, 29th March 2022 12:16:01 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Monday, 20th January 2025 1:47:22 pm
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

"""
    
city is more reliable than county/zip

Deliverable/Questionable identify as Deliverable

full_address redundant

True only: can't determine negative

UNKNOWN location_type treat as missing

version 1.0 vs 2.0 may uses same scoring systems in different segments

"""

"""
This file is for flatten data to different sources in csv format

Output sources:
1. tabular: tabular data to build models
2. NLP: text information to build language processing models
3. CV: image information to build image processing models
4. archive: data to be archived

Flatten methods (general rules):
1. dictionary: flatten by key-value pairs
2. list of dictionary: a. select certain key as column names and value as column values (may encounter different keys)
                       b. flag columns with number of items
3. list: one hot encoding
"""

import os
import psutil
import argparse
import ast
import warnings
import time
import json
import numpy as np
import pandas as pd
import itertools
from functools import partial
import multiprocessing
from multiprocessing import Pool

# from collections import defaultdict
# from tqdm.contrib.concurrent import process_map

# ignore certain warnings
# will encounter empty slice warning when get all versions scores are not available when flatten,
# but since will return nan, it's fine (suppress warning)
warnings.filterwarnings("ignore", ".*Mean of empty slice*.")
warnings.filterwarnings("ignore", category=DeprecationWarning)
####################################################################################################
# get args

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
parser.add_argument("--Nthreads", type=int, default=-1, help="number of threads")
args = parser.parse_args()

####################################################################################################
# utilities


def str2list(item):

    try:
        return ast.literal_eval(item)
    #         return json.loads(item)
    except:
        return item


def list_dict(item):

    # make sure the item is a list of dicts
    # it's not missing
    if isinstance(item, list) and item:
        if isinstance(item[0], dict):
            return True

    return False


# convert string type dict to dict type
def str2dict(item):

    try:
        return json.loads(item)
    # when facing errors, return original form
    except:
        return item


# only add mean column when more than one unqiue key
# and all of them are numerical
def whether_add_mean(_list):

    if len(_list) > 1:
        for _item in _list:
            try:
                float(_item)
            except:
                return False
        return True
    else:
        return False


# post processing for the original column
def post_original_column(item):

    if isinstance(item, dict):
        return np.nan
    else:
        return item


# combine early stop column from one row
def combine_columns(data, sep="."):

    result = ""

    for item in data:
        if not pd.isna(item):
            if not result:
                result += item
            else:
                result += sep + item

    return result


# get early stop columns
def get_early_stop(file_path, sheet_name="early_stop"):

    early_stop_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    early_stop_raw.drop("P.S.", axis=1, inplace=True)

    early_stop = []

    for index in early_stop_raw.index:
        early_stop.append(combine_columns(early_stop_raw.loc[index, :]))

    return early_stop


# get to_categorical columns
def get_to_categorical(file_path, sheet_name="to_categorical"):

    to_categorical_raw = pd.read_excel(file_path, sheet_name=sheet_name)

    to_categorical = []

    for index in to_categorical_raw.index:
        tmp = {}
        tmp["column"] = combine_columns(
            to_categorical_raw.loc[
                index, ~to_categorical_raw.columns.isin(["key", "value"])
            ]
        )
        tmp["key"] = to_categorical_raw.loc[index, "key"]
        tmp["value"] = to_categorical_raw.loc[index, "value"]
        to_categorical.append(tmp)

    to_categorical_column = [item["column"] for item in to_categorical]

    return to_categorical, to_categorical_column


# get special methods
def get_special(file_path, sheet_name="special"):

    special_cases_raw = pd.read_excel(file_path, sheet_name=sheet_name)

    special_cases = []

    for index in special_cases_raw.index:
        tmp = {}
        tmp["column"] = combine_columns(
            special_cases_raw.loc[
                index, ~special_cases_raw.columns.isin(["key", "value", "method"])
            ]
        )
        tmp["key"] = special_cases_raw.loc[index, "key"]
        tmp["value"] = special_cases_raw.loc[index, "value"]
        tmp["method"] = special_cases_raw.loc[index, "method"]
        special_cases.append(tmp)

    special_cases_column = [item["column"] for item in special_cases]

    return special_cases, special_cases_column


# find standard columns and their output_path
def get_output(file_path, sheet_name="output"):

    output = pd.read_excel(file_path, sheet_name=sheet_name)
    output = output.loc[:, output.columns.isin(["name", "output_path"])]

    return output


def get_dictInlist(item, error="mapping"):

    if isinstance(item, list) and item:
        if isinstance(item[0], dict):
            if len(item) > 1:
                if error == "mapping":
                    result = {}
                    # use first item as key index
                    # how to make sure all keys are included ???
                    unique_keys = set().union(*[_item.keys() for _item in item])
                    for key in unique_keys:
                        result[key] = list(
                            _item[key] if key in _item.keys() else np.nan
                            for _item in item
                        )
                    return result
                elif error == "raise":
                    raise ValueError("Get more than 1 item!")
                elif error == "warn":
                    warnings.warn("Get more than 1 item, returns the first one.")
                    return item[0]
            else:
                return item[0]

    return item


def extract_order_value(item, order, key="facet", value="score"):

    if not isinstance(order, np.ndarray):
        order = np.array(order)
    result = np.empty(len(order))
    result[:] = np.nan

    if isinstance(item, list):
        for _item in item:
            # if find item with no key/value, pass
            try:
                result[np.where(order == _item[key])[0][0]] = _item[value]
            except:
                pass

    return result


def check_for_key(data, key="version", error="mapping", verbose=False):
    """
    check whether key is unique in one column,
    if not unique, inform users
    """

    unique_key = []

    for item in data:
        # list of items
        if isinstance(item, list):
            for _item in item:
                if key in _item.keys():  # make sure key in _item
                    if _item[key] not in unique_key:
                        unique_key.append(_item[key])

    if error == "mapping":

        if verbose:
            print(
                "[INFO] Unique {} found in column {:50s} are: {}.\n".format(
                    key, data.name, ", ".join(unique_key)
                )
            )
        else:
            pass
    elif len(unique_key) > 1:
        if error == "warn":

            warnings.warn("More than one unique key, mapping to list of values.")
        elif error == "raise":

            raise ValueError("Get more than one unique key!")

    return unique_key


def to_category(data, key="facet", value="score", sep="."):
    """
    list of dictionary, in form of key and value
    convert to categorical columns (key as column head)
    """

    prefix = data.name  # get prefix

    # get all unique keys
    unique_keys = check_for_key(data, key)

    # create categorical flatten
    result = pd.DataFrame(columns=unique_keys)

    for item in data:
        if isinstance(item, list):
            tmp_row = {key: [], value: []}

            for _item in item:
                if key in _item.keys():  # make sure key in the item keys
                    if _item[key] not in tmp_row[key]:
                        tmp_row[key].append(_item[key])
                        if value in _item.keys():  # make sure value in the item keys
                            tmp_row[value].append(_item[value])
                        else:  # if not exists, use nan as placeholder
                            tmp_row[value].append(np.nan)
            #             can get duplicate key/value pairs, below method can not handle
            #             tmp_row = defaultdict(list)
            #             {tmp_row[_key].append(sub[_key]) for sub in item for _key in sub}
            #             tmp_row = dict(tmp_row) # force to dict type
            # convert to pandas series for combining

            # if get empty key/value
            if len(tmp_row[value]) > 0:
                tmp_row = pd.Series(tmp_row[value], index=tmp_row[key])
                result = result.append(tmp_row, ignore_index=True)
            else:
                result = result.append(pd.Series(), ignore_index=True)
        else:
            # add empty row
            result = result.append(pd.Series(), ignore_index=True)
            # need empty row to fit size
    #             tmp_row = np.empty(len(unique_keys))
    #             tmp_row[:] = np.nan
    #             tmp_row = pd.Series(tmp_row, index = unique_keys)
    #             result = result.append(tmp_row, ignore_index = True)

    result = result.add_prefix(prefix + sep)  # add prefix
    result.index = data.index  # force uniform index

    #     print("[INFO] Column {:50s} flatten to categorical variables.".format(prefix))

    return result


# flag column as True/False
def flag_data(data, key=None, value=None, suffix="count", sep="_"):

    # new format for len function
    def _len(item):
        try:
            return len(item)
        except:
            return item

    # if count, count items in the cell
    if suffix == "count":

        #         print("[INFO] Column {:50s} flatten as flagged variable.".format(data.name))
        result = data.apply(lambda x: _len(x))
    # if bool, only check whether empty
    elif suffix == "bool":

        #         print("[INFO] Column {:50s} flatten as flagged variable.".format(data.name))
        result = data.notnull()

    # add suffix to the column name
    result.name += sep + suffix
    return pd.concat([data, result], axis=1)


def version_to_categorical(
    data, key="version", value=["score"], sep=".", error="mapping"
):

    # force value to be a list
    if not isinstance(value, list) and not pd.isna(value):
        # if only string (no list brackets), add brackets
        if isinstance(value, str) and value[0] != "[":
            value = [value]
        else:
            value = ast.literal_eval(value)

    prefix = data.name  # find prefix
    # find unique_keys
    unique_key = check_for_key(data, key, error)

    # possible column names for multiple value
    column_names = [
        _value + "_" + key + "_" + item for item in unique_key for _value in value
    ]
    result = pd.DataFrame(columns=column_names)

    for item in data:
        if isinstance(item, list):  # in case facing nan or not properly-structured item
            tmp_row = {key: []}
            for _value in value:
                tmp_row[_value] = []

            for _item in item:
                if key in _item.keys():  # make sure key in the item keys
                    if _item[key] not in tmp_row[key]:
                        tmp_row[key].append(_item[key])
                        for _value in value:
                            if (
                                _value in _item.keys()
                            ):  # make sure value in the item keys
                                tmp_row[_value].append(_item[_value])
                            else:  # if missing, use nan as placeholder
                                tmp_row[_value].append(np.nan)
            #             can get duplicate/missing key/value pairs, below method can not handle
            #             tmp_row = defaultdict(list)
            #             {tmp_row[_key].append(sub[_key]) for sub in item for _key in sub}
            #             tmp_row = dict(tmp_row) # force to dict type
            # only valid when find any keys
            if len(tmp_row[key]) > 0:
                tmp_value = [tmp_row[_value] for _value in value]
                tmp_name = [
                    _value + "_" + key + "_" + item
                    for item in tmp_row[key]
                    for _value in value
                ]
                # convert to pandas series for combining
                tmp_row = pd.Series(np.reshape(tmp_value, (1, -1))[0], index=tmp_name)
                # put values back
                result = result.append(tmp_row, ignore_index=True)
            else:
                result = result.append(pd.Series(), ignore_index=True)
        #             for _item in item :
        #                 for idx, _value in enumerate(value) :
        #                     try :
        #                         tmp[idx, unique_key.index(_item[key])] = _item[_value]
        #                     # at some cases, no key/value found in data
        #                     except :
        #                         pass
        else:
            # add empty row
            result = result.append(pd.Series(), ignore_index=True)
    #             tmp_row = np.empty(len(np.reshape(column_names, (1, -1))[0]))
    #             tmp_row[:] = np.nan
    #             # convert to pandas series for combining
    #             tmp_row = pd.Series(tmp_row, index = np.reshape(column_names, (1, -1))[0])
    #             # put values back
    #             result = result.append(tmp_row, ignore_index = True)
    # put values back
    #         tmp = pd.Series(
    #             np.reshape(tmp, (1, -1))[0], index = np.reshape(column_names, (1, -1))[0]
    #         )
    #         result = result.append(tmp, ignore_index = True)

    # add mean column, for all
    # only add mean columns if there's multiple unique keys
    if whether_add_mean(unique_key):
        for _value in value:
            # mean should based only on the specific value
            result[_value + "_" + key + "_mean"] = np.nanmean(
                result.filter(regex=_value), axis=1
            )

    result = result.add_prefix(prefix + sep)  # add prefix
    result.index = data.index  # force uniform index

    #     print("[INFO] Column {:50s} flatten to categorical variables by version.".format(prefix))

    # put original data back, for archive purpose
    return pd.concat([data, result], axis=1)


# one hot encoding columns in three cases
# 1. all valid values are lists
# 2. partial valid values are lists
# 3. non of valid values are lists
def cased_one_hot(data, key=None, value=None, sep="."):

    prefix = data.name  # find prefix

    # select only valid values
    data = data[data.notnull()]

    # case 1 or 2
    case_crit = data.apply(lambda x: isinstance(x, list))
    if case_crit.any():

        # if case 2, convert to 1 before one hot encoding
        if not case_crit.all():
            data = data.apply(lambda x: [x] if not isinstance(x, list) else x)

        # now, all valid values are lists
        # use MultiLabelBinarizer
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer()

        return pd.DataFrame(
            mlb.fit_transform(data), columns=mlb.classes_, index=data.index
        ).add_prefix(
            prefix + sep
        )  # add prefix
    # case 3
    else:
        return pd.get_dummies(data, prefix=prefix)


# special method mapping
method_mapping = {
    "flag": flag_data,
    "to category": to_category,
    "version to categorical": version_to_categorical,
    "cased one hot": cased_one_hot,
}


####################################################################################################
# flatten


def cased_flatten(data, stored_data=pd.DataFrame()):

    _column = data.name  # find name for cased flatten

    if _column in early_stop:

        #         print("Column {:50s} flattened.".format(_column))

        return pd.concat([stored_data, data], axis=1)
    elif _column in special_cases_column:

        # find key/value for the special case
        for item in special_cases:
            if item["column"] == _column:
                _method = item["method"]
                _key = item["key"]
                _value = item["value"]
                break

        # apply the function, put into stored_data
        # the function contains processing methods
        return pd.concat(
            [stored_data, method_mapping[_method](data, key=_key, value=_value)], axis=1
        )
    # decrepted case
    # elif _column in to_categorical_column:

    #     # find the corresponding key/value for the column
    #     for item in to_categorical:
    #         if item["column"] == _column:
    #             _key = item["key"]
    #             _value = item["value"]
    #             break

    #     # apply to categorical, put into stored_data
    #     return pd.concat(
    #         [stored_data, to_category(data, key=_key, value=_value)], axis=1
    #     )
    else:

        # if not special, apply list_flatten
        #         print("Column {:50s} flattened.".format(_column))
        return pd.concat([stored_data, list_flatten(data)], axis=1)


def list_flatten(data, error="mapping"):

    # evaluate the columns
    tmp_data = data.apply(lambda x: str2list(x))
    # check whether contains list
    # only flatten when it's a list of dict
    # no need to further flatten when list of strings or others

    if tmp_data.apply(lambda x: list_dict(x)).any():
        # get first item
        tmp_data = tmp_data.apply(lambda x: get_dictInlist(x, error))
        # select dict part
        is_dict = tmp_data.apply(lambda x: isinstance(x, dict))
        dict_part = tmp_data[is_dict]

        # flatten
        _flatten_list = pd.json_normalize(dict_part).add_prefix(tmp_data.name + ".")
        _flatten_list.index = dict_part.index

        # make sure it's flatten to bottom
        flatten_list = pd.DataFrame()
        for _column in _flatten_list.columns:
            flatten_list = cased_flatten(_flatten_list[_column], flatten_list)
        #             if _column in early_stop :
        #                 flatten_list = pd.concat([flatten_list, _flatten_list[_column]], axis = 1)
        #             elif _column in to_categorical_column :
        #                 for item in to_categorical :
        #                     # find the corresponding key/value for the column
        #                     if item["column"] == _column :
        #                         _key = item["key"]
        #                         _value = item["value"]
        #                         break
        #                 flatten_list = pd.concat(
        #                     [flatten_list, to_category(
        #                         _flatten_list[_column], key = _key, value = _value
        #                     )], axis = 1
        #                 )
        #             else :
        #                 flatten_list = pd.concat(
        #                     [flatten_list, list_flatten(_flatten_list[_column])], axis = 1
        #                 )

        original_part = tmp_data[~is_dict]
        original_part.columns = tmp_data.name
        # if original all null, return only flatten part
        if original_part.isnull().all():
            return flatten_list
        else:
            return pd.concat([original_part, flatten_list], axis=1)

    return tmp_data


def dict_flatten(data):

    # convert string type dict to dict
    tmp_data = data.apply(lambda x: str2dict(x))

    # check whether contains dict
    # if not, no need for further flatten
    if tmp_data.apply(lambda x: isinstance(x, dict)).any():

        # if early stop of dict at highest level, need to decide here
        if tmp_data.name in early_stop:
            return cased_flatten(tmp_data)

        # get locations of dict
        # can not directly apply json_normalize if contains non-dict
        is_dict = tmp_data.apply(lambda x: isinstance(x, dict))
        dict_part = tmp_data[is_dict]
        # normalize only
        flatten_dict = pd.json_normalize(dict_part).add_prefix(tmp_data.name + ".")
        # set index to match
        flatten_dict.index = dict_part.index
        # # convert the original column
        # raw_data[column] = raw_data[column].apply(lambda x: post_original_column(x))
        # # put flatten column back to data
        # # at this stage, the flatten_dict will be matched by index
        # raw_data = pd.concat([raw_data, flatten_dict], axis = 1)

        flatten_result = pd.DataFrame()

        for _column in flatten_dict.columns:
            flatten_result = cased_flatten(flatten_dict[_column], flatten_result)
        #             if _column in early_stop :
        #                 flatten_result = pd.concat([flatten_result, flatten_dict[_column]], axis = 1)
        #             elif _column in to_categorical_column :
        #                 for item in to_categorical :
        #                     # find the corresponding key/value for the column
        #                     if item["column"] == _column :
        #                         _key = item["key"]
        #                         _value = item["value"]
        #                         break
        #                 flatten_result = pd.concat(
        #                     [flatten_result, to_category(
        #                         flatten_dict[_column], key = _key, value = _value
        #                     )], axis = 1
        #                 )
        #             else :
        #                 flatten_result = pd.concat(
        #                     [flatten_result, list_flatten(flatten_dict[_column])], axis = 1
        #                 )

        original_part = tmp_data[~is_dict]
        original_part.columns = tmp_data.name

        # if original all null, return only flatten part
        if original_part.isnull().all():
            return flatten_result
        else:
            return pd.concat([original_part, flatten_result], axis=1)
    else:
        tmp_data = data.apply(lambda x: str2list(x))
        if tmp_data.apply(lambda x: isinstance(x, list)).any():
            # empty stored data
            return cased_flatten(tmp_data)
    #             if tmp_data.name in early_stop :
    #                 return tmp_data
    #             elif tmp_data.name in to_categorical_column :
    #                 for item in to_categorical :
    #                     # find the corresponding key/value for the column
    #                     if item["column"] == tmp_data.name :
    #                         _key = item["key"]
    #                         _value = item["value"]
    #                         break
    #                 return to_category(tmp_data, key = _key, value = _value)
    #             else :
    #                 return list_flatten(tmp_data)

    return tmp_data


def pool_task(raw_data):

    # if os.path.exists("Split " + str(i + 1) + ".txt") :
    #     os.remove("Split " + str(i + 1) + ".txt")
    # f = open("Split " + str(i + 1) + ".txt", "w")
    # txt_write = f.write("Split {} ...\n".format(i + 1))
    # f.close() # make sure it's available for checking
    # raw_data = pd.read_csv("output_country_" + str(i + 1) + ".csv", low_memory = False)

    # start_time = time.time()
    # remove non-informal entries
    remove_index = []
    for _index in raw_data.index:
        # if only contains ["CARPE_ID", "REC_ID", "CLIENT_ID", "YEAR", "MATCH_SCORE"]
        # remove these entries
        if (
            raw_data.loc[_index, :]
            .drop(["CARPE_ID", "CYBER_COMPANY_ID", "CIK", "MATCH_SCORE"])
            .isnull()
            .all()
        ):
            #         print("Index {} should be removed".format(_index))
            remove_index.append(_index)
    raw_data.drop(index=remove_index, inplace=True)
    # f = open("Split " + str(i + 1) + ".txt", "a")
    # print(
    #     "Number of entries removed: {}.\nRemaining entires: {}.".format(
    #         len(remove_index), len(raw_data)
    #     )
    # )

    # reset index
    raw_data.reset_index(drop=True, inplace=True)
    # end_time = time.time()
    # print(
    #     "Time taken to remove of non-informal entries: {:.4f} seconds.\n".format(end_time - start_time)
    # )
    # f.close()

    # flatten process
    # start_time = time.time()
    flatten_data = pd.DataFrame()
    for _column in raw_data.columns:
        flatten_data = pd.concat(
            [flatten_data, dict_flatten(raw_data[_column])], axis=1
        )
    # end_time = time.time()
    # print(
    #     "Time taken to further flatten the data: {:.4f} seconds.\n".format(end_time - start_time)
    # )
    # f.close()

    # start_time = time.time()
    for column in flatten_data.columns:
        flatten_data[column] = flatten_data[column].apply(lambda x: unify_nan(x))
    # end_time = time.time()
    # f = open("Split " + str(i + 1) + ".txt", "a")
    # print("Time taken to unify nan: {:.4f} seconds.\n".format(end_time - start_time))
    # f.close()

    # start_time = time.time()
    flatten_data = formatting(flatten_data)
    # end_time = time.time()
    # f = open("Split " + str(i + 1) + ".txt", "a")
    # print("Time taken to format dataset: {:.4f} seconds.\n".format(end_time - start_time))
    # f.close()
    # print("Size after formatting: {}".format(flatten_data.shape))
    # flatten_data.to_csv("output_country_flatten_" + str(i + 1) + ".csv", index = False)

    return flatten_data


####################################################################################################
# post-flatten


# unify nan
# for better missing rate check
def unify_nan(
    item,
    nas=["none", "nan", "", "[]", "{}", "[{}]", ",", "na", "missing", "unknown"],
):

    # exclude upper case str scenarios
    if str(item).lower() in nas:
        return np.nan
    else:
        return item


# check if column only contain list with one value
# if yes, extract value from the list


# if all length <= 1, return True
def len_bool(data):

    for item in data:
        if isinstance(item, list):
            if len(item) > 1:
                return False
    return True


# remvoe [] if only one value available across the data
def formatting(data):

    for column in data.columns:
        if len_bool(data[column]):
            data[column] = data[column].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )

    return data


# compare flatten columns with standard columns
def check_for_columns(flatten_columns, standard_columns, error="warn"):

    # new columns - old columns
    new_minus_old = list(set(flatten_columns) - set(standard_columns))
    # old columns - new columns
    old_minus_new = list(set(standard_columns) - set(flatten_columns))

    if len(new_minus_old) > 0:
        if error == "warn":
            warnings.warn("Get unseen columns: {}.".format(",".join(new_minus_old)))
        elif error == "error":
            raise ValueError("Get unseen columns: {}.".format(",".join(new_minus_old)))

    if len(old_minus_new) > 0:
        if error == "warn":
            warnings.warn("Missing columns: {}.".format(",".join(old_minus_new)))
        elif error == "error":
            raise ValueError("Missing columns: {}.".format(",".join(old_minus_new)))


# in case multiple output source for one type
# find all unique sources
def _output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format, type):

    # find unique output sources under same type
    unique_source = pd.unique(
        output_format.loc[
            output_format["output_path"].str.contains(type), "output_path"
        ]
    )

    # make sure the folder exists
    if not os.path.exists(os.path.join(OUTPUT_PATH, type)):
        os.makedirs(os.path.join(OUTPUT_PATH, type))

    # output each source to one file
    for _source in unique_source:
        # slice data by output_source
        # export to corresponding folder and files
        _path = os.path.join(
            OUTPUT_PATH, type, OUTPUT_FILE.replace(".csv", "") + "_" + _source + ".csv"
        )

        # only write entries containing valid values, to save storage
        type_columns = output_format.loc[
            output_format["output_path"].isin([_source]), "name"
        ]
        select_index = flatten_data[type_columns].notnull().any(axis=1)

        # include prefix for output
        output_columns = output_format.loc[
            output_format["output_path"].isin(["prefix", _source]), "name"
        ]

        flatten_data.loc[select_index, output_columns].to_csv(_path, index=False)


# output to sources by format
def output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format):

    # output to tabular source
    _output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format, "tabular")
    # output to NLP source
    _output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format, "NLP")
    # output to CV source
    _output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format, "CV")
    # output to archive source
    _output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format, "archive")


if __name__ == "__main__":

    # initial RAM usage
    print(
        "Initial RAM usage for data flatten:\nUsed RAM: {:.4f} GB, Total RAM: {:.4f} GB.\n".format(
            psutil.virtual_memory().used / (2**30),
            psutil.virtual_memory().total / (2**30),
        )
    )

    # get args
    SOURCE_FILE = args.source_file
    OUTPUT_PATH = args.output_path
    OUTPUT_FILE = args.output_file
    FORMAT_TABLE = args.format_table
    # get number threads
    # if -1, get all available threads
    NTHREAD = multiprocessing.cpu_count() if args.Nthreads == -1 else args.Nthreads

    # get early stop columns
    early_stop = get_early_stop(FORMAT_TABLE, sheet_name="early_stop")

    # decrepted case
    # # get to categorical columns
    # to_categorical, to_categorical_column = get_to_categorical(
    #     FORMAT_TABLE, sheet_name="to_categorical"
    # )

    # get special methods
    special_cases, special_cases_column = get_special(
        FORMAT_TABLE, sheet_name="special"
    )

    start_time = time.time()

    print("Reading data ...\n")

    # read data
    if SOURCE_FILE[-3:] == "csv":
        raw_data = pd.read_csv(SOURCE_FILE, low_memory=False)
    elif SOURCE_FILE[-4:] == "json":
        raw_data = pd.read_json(SOURCE_FILE)

    print("Data read in.\n")

    # RAM usage
    print(
        "RAM usage after reading data:\nUsed RAM: {:.4f} GB, Total RAM: {:.4f} GB.\n".format(
            psutil.virtual_memory().used / (2**30),
            psutil.virtual_memory().total / (2**30),
        )
    )

    print("Pool tasks begin ...")
    # split index by threads
    divide_index = np.array_split(raw_data.index, NTHREAD)

    # pooling flatten process
    pool = Pool(NTHREAD)
    flatten_data_list = pool.map(
        pool_task, [raw_data.loc[_index] for _index in divide_index]
    )
    pool.close()
    pool.join()

    print("Pool tasks completed.\n")

    # RAM usage
    print(
        "RAM usage after pool task:\nUsed RAM: {:.4f} GB, Total RAM: {:.4f} GB.\n".format(
            psutil.virtual_memory().used / (2**30),
            psutil.virtual_memory().total / (2**30),
        )
    )

    print("Combine pool results ...")

    flatten_data = pd.concat(flatten_data_list, axis=0)

    # special treatment for the combined data

    # 1. for columns from cased one hot, need to fill na with 0
    # cause: some pool tasks may not have seen the column
    fillna_columns = []
    # find all original columns with cased one hot encoding (as rule)
    for item in special_cases:
        if item["method"] == "cased one hot":
            fillna_columns.append(item["column"])

    for column_rule in fillna_columns:
        # find all flattend columns generated from the rule
        for column in flatten_data.columns[
            flatten_data.columns.str.contains(column_rule)
        ]:
            flatten_data[column].fillna(0, inplace=True)

    # preserve index at above step (for order), reset below
    flatten_data.reset_index(drop=True, inplace=True)
    end_time = time.time()
    print("flatten_data size: {}\n".format(flatten_data.shape))

    fix_columns = ["CARPE_ID", "CYBER_COMPANY_ID", "CIK", "MATCH_SCORE"]
    # sort the columns
    column_list = list(set(flatten_data.columns) - set(fix_columns))
    column_list.sort()

    print("Pool results combined.\n")

    # RAM usage
    print(
        "RAM usage after pool data combined:\nUsed RAM: {:.4f} GB, Total RAM: {:.4f} GB.\n".format(
            psutil.virtual_memory().used / (2**30),
            psutil.virtual_memory().total / (2**30),
        )
    )

    # check for columns whether getting unseen columns or missing columns
    output_format = get_output(FORMAT_TABLE, sheet_name="output")

    check_for_columns(list(flatten_data.columns), output_format["name"])

    print("Write to file ...\n")

    # output flatten data
    # flatten_data[fix_columns + column_list].to_csv(OUTPUT_FILE, index=False)

    # make sure all flatten columns are seen before
    # in case use warning and get here
    output_format = output_format.loc[
        output_format["name"].isin(list(flatten_data.columns)), :
    ]

    # output sources: tabular, NLP, CV, archive
    output_to_source(flatten_data, OUTPUT_PATH, OUTPUT_FILE, output_format)

    print("All write to file. All tasks completed.\n")

    print("Total time taken: {:.4f} seconds.".format(end_time - start_time))
