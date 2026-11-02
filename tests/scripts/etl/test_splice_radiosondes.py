# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.etl.nnja.conv.splice_radiosondes import read_messages, splice


def bufr(category, payload, subsets=1, edition=4, section2=b""):
    if edition >= 4:
        section1 = bytearray(22)
        section1[10] = category
        section1[9] = 0x80 if section2 else 0
    else:
        section1 = bytearray(18)
        section1[8] = category
        section1[7] = 0x80 if section2 else 0
    section1[:3] = len(section1).to_bytes(3, "big")
    if section2:
        section2 = (4 + len(section2)).to_bytes(3, "big") + b"\x00" + section2
    section1 = bytes(section1) + section2
    section3 = bytearray(9)
    section3[:3] = len(section3).to_bytes(3, "big")
    section3[4:6] = subsets.to_bytes(2, "big")
    section4 = (4 + len(payload)).to_bytes(3, "big") + b"\x00" + payload
    length = 8 + len(section1) + len(section3) + len(section4) + 4
    message = (
        b"BUFR"
        + length.to_bytes(3, "big")
        + bytes([edition])
        + section1
        + section3
        + section4
        + b"7777"
    )
    return message + b"\x00" * (-len(message) % 8)


def test_splice_replaces_only_the_radiosonde_block():
    table = bufr(11, b"table")
    noaa = (
        table
        + bufr(102, b"short", 3)
        + bufr(102, b"short2", 2)
        + bufr(104, b"aircraft", edition=3)
    )
    prepnr = table + bufr(102, b"long-sounding", 5) + bufr(105, b"other")

    data, old, new = splice(noaa, prepnr)

    messages = read_messages(data)
    assert (old, new) == (2, 1)
    assert [(m.category, m.subsets) for m in messages] == [(11, 1), (102, 5), (104, 1)]
    assert data == table + bufr(102, b"long-sounding", 5) + bufr(
        104, b"aircraft", edition=3
    )
    assert all(len(m.padded) % 8 == 0 for m in messages)


def test_subsets_are_read_past_an_optional_section_2():
    message = bufr(102, b"x", subsets=7, section2=b"local")
    assert read_messages(message)[0].subsets == 7


def test_splice_refuses_mismatched_tables_and_bad_padding():
    noaa = bufr(11, b"table") + bufr(102, b"a")
    with pytest.raises(ValueError, match="table"):
        splice(noaa, bufr(11, b"other table") + bufr(102, b"b"))
    with pytest.raises(ValueError, match="padding"):
        read_messages(noaa + b"\x01")
    with pytest.raises(ValueError, match="fewer radiosonde"):
        splice(noaa, bufr(11, b"table") + bufr(104, b"aircraft"))
