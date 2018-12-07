import pytest
from deep_abyasa import Utils
from deep_abyasa import Encode_Labels
from deep_abyasa import CustomException


@pytest.fixture
def default_labels():
    return Utils.read_json("./deep_abyasa/tests/data/sample.json")['label']

@pytest.fixture
def default_multi_labels():
    return Utils.read_json("./deep_abyasa/tests/data/sample2.json")['label']

@pytest.fixture
def multi_labels_delimiter():
    return Utils.read_json("./deep_abyasa/tests/data/sample3.json")['label']


def test_determine_read_file_method_json():
    func = Encode_Labels.determine_read_file_method('json')
    assert(type(func) is type(Utils.read_json))


def test_determine_read_file_method_something_else():
    with pytest.raises(CustomException):
        Encode_Labels.determine_read_file_method('gibrish')


def test_determine_label_extract_method_single_labels():
    func = Encode_Labels.determine_label_extract_method(False, None)
    assert(type(func) is type(Encode_Labels.extract_single_labels))


def test_determine_label_extract_method_multi_labels_arr_like():
    func = Encode_Labels.determine_label_extract_method(True, None)
    assert(type(func) is type(Encode_Labels.extract_multi_labels_from_arraylike_cols))


def test_determine_label_extract_method_multi_labels_delimited():
    func = Encode_Labels.determine_label_extract_method(True, ";")
    assert(type(func) is type(Encode_Labels.extract_multi_labels_from_delimited_cols))


@pytest.fixture
def test_extract_single_labels(default_labels):
    labels = sorted(Encode_Labels.extract_single_labels(default_labels))
    actuals = ['f1', 'f2', 'f2e']
    assert labels == actuals
    return labels


def test_extract_multi_labels_from_arraylike_cols(default_multi_labels):
    labels = sorted(Encode_Labels.extract_multi_labels_from_arraylike_cols(default_multi_labels))
    actuals = ['f1', 'f2', 'f2e', 'f3', 'f4' ]
    assert labels == actuals


def test_extract_multi_labels_from_delimited_cols(multi_labels_delimiter):
    labels = sorted(Encode_Labels.extract_multi_labels_from_delimited_cols(multi_labels_delimiter, ";"))
    actuals = ['f1', 'f2', 'f2e', 'f3', 'f4']
    assert labels == actuals


def test_generate_itol_ltoi(test_extract_single_labels):
    actual_itol = {0: 'f1', 1: 'f2', 2: 'f2e'}
    itol, ltoi = Encode_Labels.generate_itol_ltoi(test_extract_single_labels)
    assert(sorted(itol.keys()) == sorted(actual_itol.keys()))
    assert (sorted(itol.values()) == sorted(actual_itol.values()))
    assert (sorted(ltoi.values()) == sorted(actual_itol.keys()))
    assert (sorted(ltoi.keys()) == sorted(actual_itol.values()))


def test_extract_labels_from_index_files_determine_files():
    itol, ltoi = Encode_Labels.encode_from_index_files('./deep_abyasa/tests/data', "label", files=None,
                                                               file_type='json', pattern="multi[0-9].json",
                                                               multi_label=True, multi_label_delimiter=None)
    actual_itol = {0: "f1", 1: "f2", 2: "f2e", 3: "f3", 4: "f4", 5: "f4e", 6: "f5", 7: "f8"}
    assert (sorted(itol.keys()) == sorted(actual_itol.keys()))
    assert (sorted(itol.values()) == sorted(actual_itol.values()))


def test_extract_labels_from_index_files_from_list():
    itol, ltoi = Encode_Labels.encode_from_index_files('./deep_abyasa/tests/data', "label",
                                                               files=["sample.json", "sample3.json"],
                                                               file_type='json', multi_label=True,
                                                               multi_label_delimiter=";")
    actual_itol = {0: "f1", 1: "f2", 2: "f2e", 3: "f3", 4: "f4"}
    assert (sorted(itol.keys()) == sorted(actual_itol.keys()))
    assert (sorted(itol.values()) == sorted(actual_itol.values()))


def test_encode_from_pickle_keys():
    itol, ltoi = Encode_Labels.encode_from_pickle("./deep_abyasa/tests/data/periodic_table_elements.pkl")
    assert (itol[0] == 'Ac')
    assert (ltoi['Ac'] == 0)


def test_encode_from_pickle_values():
    itol, ltoi = Encode_Labels.encode_from_pickle("./deep_abyasa/tests/data/periodic_table_elements.pkl", key=False)
    assert (itol[0] == 'actinium')
    assert (ltoi['actinium'] == 0)

