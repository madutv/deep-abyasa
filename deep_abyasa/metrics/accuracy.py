from mxnet.metric import EvalMetric, check_label_shapes
from mxnet import ndarray
import numpy as np


class AccuracyMultiLabel(EvalMetric):
    """Child class of mxnet EvalMetric to calculate accuracy metric for
       multi-lable datasets

    """
    def __init__(self, axis=1, name='accuracy_multi',
                 output_names=None, label_names=None):
        super(AccuracyMultiLabel, self).__init__(name, axis=axis,
                                             output_names=output_names,
                                             label_names=label_names)
        self.axis = axis
        self.pred_status = {}

    def update(self, labels, preds):
        """Implementation of update method. Updates accuracy: sum_metric
           and num_inst. Item is treated as accurate if all the labels
           from prediction matches with all the given labels

           Args:
               lables: Acutals

               preds: Predication
        """
        labels, preds = check_label_shapes(labels, preds, True)
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            zipped = zip(pred_label, label)
            zumba = [np.all((a == b)) for a, b in zipped]
            #print(f'pred_label: {len(pred_label)}')
            self.sum_metric += sum(zumba)
            #print(f'sum_metric: {self.sum_metric}')
            self.num_inst += len(pred_label)

    def get_incorrect_preds(self, labels, preds, names):
        """Method to get incorrect predictions. Incorrect predictions
           are updated on self.pred_status attribute

        """
        labels, preds = check_label_shapes(labels, preds, True)
        #print(f'labels ahape {len(labels)}, {len(preds)}, {len(names)}')

        for label, pred_label, name in zip(labels, preds, names):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            zipped = zip(pred_label, label, name)
            for p, l, n in zipped:
                if np.all(p == l) is np.bool_(False):
                    #print(f'***** p: {p} l: {l} n: {n} ********')
                    self.pred_status[n.asscalar().astype('int32')] = [p, l]






#strict = AccuracyStrict()
#strict.update_and_get_preds(mx.nd.array([[1, 2, 3], [4, 5, 8], [4, 5, 8]]), mx.nd.array([[1, 2, 3], [4, 5, 6], [4, 5, 8]]), ['a', 'b', 'c'])
#print(f'update_and_get_preds: {strict.pred_status}')