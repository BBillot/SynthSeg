from simple_parsing.helpers.serialization import register_decoding_fn
from typing import Optional, Union, List


def decode_boolean_float_or_list(raw_value):
    if isinstance(raw_value, (float, bool, list)):
        return raw_value
    else:
        raise RuntimeError(f"Could not decode JSON value {raw_value}")


def decode_float_str_bool(raw_value):
    if isinstance(raw_value, (float, str, bool)):
        return raw_value
    else:
        raise RuntimeError(f"Could not decode JSON value {raw_value}")


def decode_none_str_int_list(raw_value):
    if isinstance(raw_value, (int, str, list)):
        return raw_value
    elif raw_value is None:
        return None
    else:
        raise RuntimeError(f"Could not decode JSON value {raw_value}")


def decode_none_str_list(raw_value):
    if isinstance(raw_value, (str, list)):
        return raw_value
    elif raw_value is None:
        return None
    else:
        raise RuntimeError(f"Could not decode JSON value {raw_value}")


register_decoding_fn(Union[bool, float, List[float]], decode_boolean_float_or_list)
register_decoding_fn(Union[bool, float, str], decode_float_str_bool)
register_decoding_fn(Union[None, str, int, List[int]], decode_none_str_int_list)
register_decoding_fn(Union[None, str, List[int]], decode_none_str_list)
