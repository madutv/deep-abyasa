from deep_abyasa import Utils


def test_create_list_of_file_paths():
    items = Utils.create_list_of_file_paths("/Users/temp", files=['world', 'peace'])
    assert('/Users/temp/world' in items)
    assert('/Users/temp/peace' in items)


def test_create_list_of_file_paths_from_pattern():
    items = Utils.create_list_of_file_paths("./", pattern=".*md")
    assert('./README.md' in items)


def test_convert_list_to_dict():
    items = Utils.convert_list_to_dict(['oxygen', 'carbon', 'hydrogen'])
    assert(items[0] == 'carbon')
    assert(items[1] == 'hydrogen')
    assert(items[2] == 'oxygen')


def test_reserve_dict():
    items = Utils.reserve_dict({0: "carbon", 1: "hydrogen", 2: "oxygen"})
    assert(items['carbon'] == 0)
    assert(items['hydrogen'] == 1)
    assert(items['oxygen'] == 2)