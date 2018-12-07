import mxnet as mx
from deep_abyasa import AccuracyMultiLabel


def test_accuracy_multi():
    metric = AccuracyMultiLabel()
    assert(metric.sum_metric == 0)
    assert(metric.num_inst == 0)

def test_accuracy_multi_update():
    metric = AccuracyMultiLabel()
    labels = mx.nd.array([[0, 1, 1], [0, 1, 0]])
    preds = mx.nd.array([[0, 1, 0], [0, 1, 0]])
    metric.update(labels, preds)
    assert(metric.num_inst == 2)
    assert(metric.sum_metric == 1)
    metric.update(labels, preds)
    assert (metric.num_inst == 4)
    assert (metric.sum_metric == 2)


def test_accuracy_multi_get_incorrect_pred():
    metric = AccuracyMultiLabel()
    labels = mx.nd.array([[0, 1, 1], [0, 1, 0]])
    preds = mx.nd.array([[0, 1, 0], [0, 1, 0]])
    names = mx.nd.array([[1], [2]])
    metric.get_incorrect_preds(labels, preds, names)
    assert(metric.pred_status[1][0].tolist() == [0, 1, 0])
    assert (metric.pred_status[1][1].tolist() == [0, 1, 1])

