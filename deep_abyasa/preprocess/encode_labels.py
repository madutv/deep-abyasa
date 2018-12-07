import pandas as pd
import numpy as np
import pickle
from deep_abyasa import Utils
from deep_abyasa import CustomException


class Encode_Labels:
    """This class contains static methods that extracts labels and returns
    dictionary object of lables to integers and integers to labels.

    Example:
         if labels are 'carbon', 'hydrogen' and 'oxygen' in index dataset
         file, then Encode_Labels.encode_from_index_files returns the
         below 2 objects:

         ::

            itol
            {
                0: carbon,
                1: hydrogen,
                2: oxygen
            }

            ltoi
            {
                carbon: 0,
                hydrogen: 1,
                oxygen: 2
            }


    """
    @staticmethod
    def encode_from_index_files(root, label_col,
                                files=None, file_type='json',
                                pattern=".*json", multi_label=False,
                                multi_label_delimiter=None):
        """Creates ltoi and itol from dataset index.

        Args:
            root: root folder where dataset index files are located

            label_col: Column that contains lables

            files: List of dataset files. If this is not provided,
                any file in root folder that matches the pattern
                will be read

            file_type: Type of index file. Default is json

            pattern: regex to match if the files list is not provided.
                If file list is provided, that takes the precedence
                and pattern will be ignored

            multi_label: Indicates whether the labels in multi-label

            multi_lable_delimiter: If the labels are multi-lable, but
                are not represented as an array type, then this parameter
                stores the delimiter.

        Returns:
            Returns a pair of dict: One going from label to int and
                another going from int ot label.

        """
        file_lists = Utils.create_list_of_file_paths(root, files, pattern)
        file_read_method = Encode_Labels.determine_read_file_method(file_type)
        label_extract_method = Encode_Labels.determine_label_extract_method(multi_label, multi_label_delimiter)
        dataframes = pd.concat([file_read_method(f) for f in file_lists])
        labels = label_extract_method(dataframes[label_col], multi_label_delimiter)
        return Encode_Labels.generate_itol_ltoi(labels)

    @staticmethod
    def encode_from_pickle(file_path, key=True):
        """This is a helper method, if lables are to be constructed from
           a pre-defined pickle file instead of deducing it from datasets.
           This one extracts labels from dict

           Args:
               file_path: Path of pickle file

               key: If true, keys will be used to calculate ltoi and itol
                    else, values are used.
        """

        try:
            elements = pickle.load(open(file_path, 'rb'))
            if key:
                return Encode_Labels.generate_itol_ltoi(list(elements.keys()))
            else:
                return Encode_Labels.generate_itol_ltoi(list(set(elements.values())))
        except Exception as e:
            pass

    @staticmethod
    def determine_read_file_method(file_type):
        """Determines method for reading dataset index files.

        Args:
            file_type: Indicate type of file. Currently only
                json method is implemented

        Returns:
            A function that reads index datasets

        """
        if file_type is 'json':
            return Utils.read_json
        else:
            print("Currently, only json index_file is implemented")
            raise CustomException

    @staticmethod
    def determine_label_extract_method(multi_label, multi_label_delimiter):
        """Determines the method for extracting lables from index files

        Args:
            multi_label: True of False, indicating if the label column is
                multi-labeled

            multi_label_delimiter: delimiter if the labels are not
                in an array type object

        Return:
            Returns a function that can extract columns

        """
        if multi_label is False:
            return Encode_Labels.extract_single_labels
        elif multi_label_delimiter is None:
            return Encode_Labels.extract_multi_labels_from_arraylike_cols
        else:
            return Encode_Labels.extract_multi_labels_from_delimited_cols

    @staticmethod
    def extract_single_labels(labels, multi_label_delimiter=None):
        """Method to extract labels when dealing with single labled
           dataset

           Args:
               labels: List of labels

               multi_label_delimiter: Not really used in this
               method, but kept for compatibility with sister
               methods

            Returns: List of unique labels

        """
        return list(labels.unique())

    @staticmethod
    def extract_multi_labels_from_arraylike_cols(labels, multi_label_delimiter=None):
        """Extract labels from multi label when the label column is array like

            Args:
                labels: List of List of lables

                multi_label_delimiter: Not really used in this method,
                    but kept for compatibility with sister methods

            Returns: List of unique labels

        """
        return list(set(np.hstack(labels)))

    @staticmethod
    def extract_multi_labels_from_delimited_cols(labels, multi_label_delimiter=None):
        """Extract labels from multi label where the labels are delimited
            by some delimiter.

            Args:
                labels: List of labels where each label is delimited
                    by something
                multi_label_delimiter: Delimiter of multi-label

            Returns: List of unique labels

        """
        lab = [list(map(lambda a: a.strip(), l.split(multi_label_delimiter))) for l in labels]
        return Encode_Labels.extract_multi_labels_from_arraylike_cols(lab)

    @staticmethod
    def generate_itol_ltoi(labels):
        """Method that generates itol and ltoi

        Args:
            labels: List of labels

        Returns:
            pair of dict: itol, ltoi

        """
        itol = Utils.convert_list_to_dict(labels)
        ltoi = Utils.reserve_dict(itol)
        return itol, ltoi