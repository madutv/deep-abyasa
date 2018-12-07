import os
import pandas as pd
import mxnet as mx
from mxnet.gluon.data import Dataset
from deep_abyasa import CustomException
from deep_abyasa import Encode_Labels


class JsonIndexMultiLabelDataset(Dataset):
    """Create mxnet Dataset for multilable from json index file
       where the Xs are links to images and Ys are array of lables.

       Args:
            root: Root path for index and images

            file: File name of index json file

            image_path: Additional path from root to get to images

            x_col: Column name that contains image details

            y_col: Column name for lables

            transform: mxnet Transformations to be applied on images

            one_hot_encode_lables: Bool indicating if the labels
                should be one hot encoded. Default is False

            determine_labels_from_y_col: If true, classes will be
                determined from lables. Default is False. This is
                used during one-hot-encoding

            labels: dict: Alternative to determine_labels_from_y_col.
                If this is provided, then it will be used for one
                hot encoding.


    """
    def __init__(self, root, file, image_path, x_col, y_col, transform=None,
                 one_hot_encode_labels=False, determine_labels_from_y_col=False,
                 labels={}):

        self.root = root
        self.file = file
        self.image_path = image_path
        self.x_col = x_col
        self.y_col = y_col
        self._transform = transform
        self.one_hot_encode_labels = one_hot_encode_labels
        self.determine_labels_from_y_col = determine_labels_from_y_col
        self.data_index = pd.read_json(os.path.join(root, file))
        self.labels = self.validate_and_set_labels(labels)

    def validate_and_set_labels(self, labels):
        """If one_hot_encode_label is set to true, this method, extracts
           labels either from index files or labels provided. If
           labels are provided that takes a precedence.

           Args:
               labels: Either labels or None. If none, labels are
                    determined from index file

           Returns:
               dict of label to int


        """
        if self.one_hot_encode_labels and labels:
            print("Since Labels are provided, this will be used for one hot encoding")
            return labels
        elif self.one_hot_encode_labels and self.determine_labels_from_y_col:
            print("Labels will be determined from y_col in dataset and will be used for hot encoding")
            labels = Encode_Labels.extract_multi_labels_from_arraylike_cols(self.data_index[self.y_col])
            itol, ltoi = Encode_Labels.generate_itol_ltoi(labels)
            return ltoi
        elif self.one_hot_encode_labels:
            print("Either Labels must be provided or determine_labels_from_y_col must be true")
            raise CustomException

    def __getitem__(self, idx):
        """Implementation of mxnet Datasets getitem method. For a given index, this
        method, read the image, applies transformations if provided, extracts
        labels, one_hot_encodes (if asked). This method also returns the
        image_name. Here the assumption is image name is of the format int.png.
        It will fail if image names are not in this format.

        Args:
            idx: Index of element to retrieve

        Returns:
            image, lables, image_name

        TODO: Fix when image_name is not of the format int.png


        """
        item = self.data_index.iloc[idx]
        image = mx.image.imread(os.path.join(self.root, self.image_path, item[self.x_col]))
        image_name = float(item[self.x_col][:-4])
        if self.one_hot_encode_labels:
            lab = [self.labels[i] for i in item[self.y_col]]
            codes = mx.nd.zeros(len(self.labels))
            for l in lab:
                codes[l] = 1
        else:
            codes = item[self.y_col]

        if self._transform is not None:
            return self._transform(image).reshape(3, image.shape[0], -1), codes, image_name
        return image.reshape(3, image.shape[0], -1), codes, image_name

    def __len__(self):
        """Implementation of mxnet len method. Simply returns number of
           elements in the dataset
        """
        return self.data_index.shape[0]
