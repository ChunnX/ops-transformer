from qliv2_test_utils import QliV2CaseSelector, QliV2ResultWriter


def test_parse_indexes():
    assert QliV2CaseSelector.parse_indexes("1,3-4", 5) == [1, 3, 4]


def test_case_name_is_stable():
    assert QliV2ResultWriter.case_name((), "decode B8") == "decode_B8"

