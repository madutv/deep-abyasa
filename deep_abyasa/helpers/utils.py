import os
import re
import pandas as pd


class Utils:
    """Bunch of helper functions

    """

    @staticmethod
    def create_list_of_file_paths(root, files=None, pattern=''):
        """Creates a list of file paths

        Args:
            root: root directory

            files: List of files

            pattern: If list of files is not provided,
                then, all files matching the pattern
                will be considered

        Returns:
            list of string of file paths, constructed
            out of root & files or patters.

        """

        if files is not None:
            return [os.path.join(root, f) for f in files]
        else:
            reg = re.compile(pattern, re.IGNORECASE)
            return [os.path.join(root, f) for f in filter(lambda a: reg.match(a), os.listdir(root))]

    @staticmethod
    def read_json(path):
        """Method to read json.

        Args:
            path: Path to json file

        Returns:
            Pandas dataframe of json file

        """
        try:
            return pd.read_json(path)
        except Exception as e:
            print(f'Failed to read Json {e}')
            raise e

    @staticmethod
    def convert_list_to_dict(items):
        """Converts a list of items to dict where key is the
           sorted order of item in list

        Args:
            items: list of items

        Returns:
            dict of items
        """
        return {i: l for i, l in enumerate(sorted(items))}


    @staticmethod
    def reserve_dict(dict_items):
        """Reverses the key and value in the dict

        Args:
            dict_items: dict to be reversed

        Returns:
            Reversed dict

        """
        return {v: k for k, v in dict_items.items()}