from deep_abyasa import JsonIndexMultiLabelDataset


def test_JsonIndexMultiLabelDataset_infer():
    ds = JsonIndexMultiLabelDataset("./deep_abyasa/tests/data",
                                    "chem_test_temp.json",
                                    "images", "file", "elements",
                                    one_hot_encode_labels=True,
                                    determine_labels_from_y_col=True)
    assert(ds.labels['carbon'] == 0)
    assert(ds.labels['hydrogen'] == 1)
    assert(ds.labels['oxygen'] == 3)
    assert(ds.labels['nitrogen'] == 2)

    x, y, n = ds.__getitem__(0)
    assert(x.shape == (3, 300, 300))
    assert(len(y) == 4)
    assert(y.asnumpy().tolist() == [1, 1, 0, 1])
    assert(n == 10091)
    assert(ds.__len__() == 3)



def test_JsonIndexMultiLabelDataset_given():
    ds = JsonIndexMultiLabelDataset("./deep_abyasa/tests/data",
                                    "chem_test_temp.json",
                                    "images", "file", "elements",
                                    one_hot_encode_labels=True,
                                    labels={'carbon': 0, 'hydrogen': 1, 'oxygen': 2, 'nitrogen': 3, 'gibrish': 4})
    assert(ds.labels['carbon'] == 0)
    assert(ds.labels['hydrogen'] == 1)
    assert(ds.labels['oxygen'] == 2)
    assert(ds.labels['nitrogen'] == 3)
    assert (ds.labels['gibrish'] == 4)

    x, y, n = ds.__getitem__(0)
    assert(x.shape == (3, 300, 300))
    assert(len(y) == 5)
    assert(y.asnumpy().tolist() == [1, 1, 1, 0, 0])
    assert(n == 10091)
    assert(ds.__len__() == 3)

