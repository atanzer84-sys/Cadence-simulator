import logging
from openpyxl import load_workbook
from typing import Any
RowDict = dict[str, Any]


def load_excel_parameters(excel_path, target_name_user_input):
    workbook = load_workbook(excel_path, data_only=True)

    # We currently assume that the **active worksheet** is the one containing the parameters we care about.
    worksheet = workbook.active
    column_headers = [_normalize_name(cell.value) for cell in worksheet[1]] 
    logging.info("Excel column headers: %s", column_headers)

    if "pl_name" not in column_headers:
        raise ValueError("Excel file has no 'pl_name' column")

    pl_name_index = column_headers.index("pl_name")
    target_name = str(target_name_user_input).strip()
    target_name_normalized = str(target_name).casefold()
    logging.info("Searching for target: '%s'", target_name)

    matching_row_dict = None
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        # Skip completely empty rows
        if row is None:
            continue

        # Guard: rows shorter than headers: i also want to get null values for empty cells
        row = list(row) + [None] * (len(column_headers) - len(row))

        pl_name_value = row[pl_name_index]
        if pl_name_value is None:
            raise ValueError("Empty pl_name in row (end of data table); no value to look up")

        pl_name_normalized = str(pl_name_value).casefold().strip()

        # First match wins. 
        # TODO: harden later when planetary transits are a thing.
        if pl_name_normalized.startswith(target_name_normalized):
            matching_row_dict = {
                header: value
                for header, value in zip(column_headers, row)
                if header is not None
            }
            logging.info("Found matching row for target '%s':", target_name_user_input)
            break

    if matching_row_dict is None:
        raise ValueError(
            f"No target found for '{target_name_user_input}' (searched until row with empty pl_name)"
        )
    return matching_row_dict, target_name

def split_stellar_planetary_parameters(dictionary, target_name):
    stellar_parameters = {}
    planetary_parameters = {}

    planet_keys = {
        "discoverymethod",
        "scale_height_km"
    }
    for key, value in dictionary.items():
        if key.startswith("pl_") or key in planet_keys:
            planetary_parameters[key] = value
        else:
            # st_* and all other parameters go to star
            stellar_parameters[key] = value

    stellar_parameters["st_name"] = target_name

    logging.info("Stellar parameters:")
    for key, value in stellar_parameters.items():
        logging.info("  %s = %r", key, value)

    logging.info("Planetary parameters:")
    for key, value in planetary_parameters.items():
        logging.info("  %s = %r", key, value)

    return stellar_parameters, planetary_parameters


def _normalize_name(name):
    """Normalize a column header: strip whitespace and remove a leading '#'."""
    if name is None:
        return None
    name = str(name).strip()
    if name.startswith("#"):
        name = name[1:]
    return name