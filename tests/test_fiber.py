import pytest

get_full_mask_result = [
    [False False False False False False False False False]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [False False False False False False False False False]
    ]

get_center_mask_result = [
    [False False False False False False False False False]
    [False False False False False False False False False]
    [False False False False False False False False False]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True  True]
    [False False False False False False False False False]
    [False False False False False False False False False]
    [False False False False False False False False False]
    ]

get_unit_vector_result = [1. 0.]
get_unit_normal_vector_result = [1. 0.]