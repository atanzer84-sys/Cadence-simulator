# import logging

# def load_parameters(path):
#     try:

#         user_parameter = {}

#         def require_param(user_parameter, key, cast_fn=None, error_msg=None):
#             if key not in user_parameter:
#                 raise ValueError(f"Missing required parameter: {key}")

#             if cast_fn is not None:
#                 try:
#                     user_parameter[key] = cast_fn(user_parameter[key])
#                 except ValueError:
#                     raise ValueError(error_msg or f"Invalid value for {key}")

#         with open(path, "r") as f:
#             for line in f:
#                 raw = line
#                 line = line.strip()

#                 if not line or line.startswith("#"):
#                     continue

#                 if "=" not in line:
#                     raise ValueError(
#                         f"Invalid line in parameter file (expected key=value): {raw.strip()}"
#                     )
#                 key, value = line.split("=", 1)
#                 user_parameter[key.strip()] = value.strip()

#         # required: target_name
#         require_param(user_parameter, "target_name")
#         target_name = user_parameter["target_name"].strip()

#         # permissive: remove one leading quote and one trailing quote if present
#         if target_name.startswith("'") or target_name.startswith('"'):
#             target_name = target_name[1:]
#         if target_name.endswith("'") or target_name.endswith('"'):
#             target_name = target_name[:-1]

#         target_name = target_name.strip()
#         if not target_name:
#             raise ValueError("target_name must not be empty")

#         user_parameter["target_name"] = target_name

#         # required: total_observation_length_h
#         require_param(
#             user_parameter,
#             "total_observation_length_h",
#             cast_fn=float,
#             error_msg="total_observation_length_h must be a number (hours)",
#         )

#         # required exposures (must be numbers, in seconds)
#         for key in ("exposure_NUV_s", "exposure_VIS_s", "exposure_IR_s"):
#             require_param(
#                 user_parameter,
#                 key,
#                 cast_fn=float,
#                 error_msg=f"{key} must be a number (seconds)",
#             )

#         logging.info("Loaded user parameters")

#         logging.info(f"target_name = {user_parameter['target_name']}")
#         logging.info(f"total_observation_length_h = {user_parameter['total_observation_length_h']}")
#         logging.info(f"exposure_NUV_s = {user_parameter['exposure_NUV_s']}")
#         logging.info(f"exposure_VIS_s = {user_parameter['exposure_VIS_s']}")
#         logging.info(f"exposure_IR_s = {user_parameter['exposure_IR_s']}")

#         return user_parameter

#     except Exception:
#             logging.exception("Failed to load and validate parameter file: %s", path)
#             raise
