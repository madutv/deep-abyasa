import time
import os
import json
from tqdm import tqdm
import numpy as np
import mxnet as mx
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet import gluon, init
from gluoncv.model_zoo import get_model
from deep_abyasa import AccuracyMultiLabel


class TrainingHelpers:
    @staticmethod
    def train(train_dl, test_dl, model, trainer,
              loss_func, epochs=20, lr_factor=0.75,
              lr_steps=[10, 20, 30, np.inf],
              metric=AccuracyMultiLabel(), num_gpus=-1):

        ctx = TrainingHelpers.get_ctx(num_gpus)
        lr_counter = 0
        num_batch = len(train_dl)
        metric.pred_status = {}

        for epoch in range(epochs):
            if epoch == lr_steps[lr_counter]:
                trainer.set_learning_rate(trainer.learning_rate * lr_factor)
                lr_counter += 1
                print(f'Learning rate is now set to: {trainer.learning_rate}')

            tic = time.time()
            train_loss = 0
            metric.reset()

            for i, batch in tqdm(enumerate(train_dl)):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
                names = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)

                # print(f"names: {names}")
                with ag.record():
                    outputs = [model(X.astype('float32')) for X in data]
                    loss = [loss_func(yhat, y) for yhat, y in zip(outputs, label)]
                for l in loss:
                    l.backward()
                trainer.step(data[0].shape[0])
                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                outs = [o.tanh().ceil().abs() for o in outputs]
                metric.update(label, outs)
                if epoch == (epochs - 1):
                    metric.get_incorrect_preds(label, outs, names)

            _, train_acc = metric.get()
            train_loss /= num_batch
            _, val_acc = TrainingHelpers.test(test_dl, model, metric=metric, num_gpus=num_gpus)

            print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
                  (epoch, train_acc, train_loss, val_acc, time.time() - tic))

        return metric.pred_status

    @staticmethod
    def get_ctx(num_gpus):
        return [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]


    @staticmethod
    def test(data_loader, model, metric=AccuracyMultiLabel(), num_gpus=-1):
        ctx = TrainingHelpers.get_ctx(num_gpus)
        metric.reset()
        for i, batch in enumerate(data_loader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [model(X) for X in data]
            outs = [o.tanh().ceil().abs() for o in outputs]
            metric.update(label, outs)
        return metric.get()

    @staticmethod
    def predict(model, root, file, itol, transform=None, num_gpus=-1):
        ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
        image = mx.image.imread(os.path.join(root, file))
        if transform is not None:
            image = transform(image)
        preds = model(image.expand_dims(axis=0).as_in_context(ctx[0]))
        a, b = np.maximum(preds.flatten().asnumpy(), 0).nonzero()
        return [itol[i] for i in b]

    @staticmethod
    def save_model(model, file_name):
        model.save_parameters(file_name)

    @staticmethod
    def create_retrain_dataset(retrain, lookup):
        train = []
        for k, v in retrain.items():
            a, b = v[1].reshape(1, -1).nonzero()
            # print(f'v: {type(b)} b: {b}')
            train.append({'file': str(k) + '.png', 'elements': [lookup[i] for i in b.tolist()]})
        return train

    @staticmethod
    def save_retrain(item):
        json.dump(item, open('chem_retrain.json', 'w'))

    @staticmethod
    def get_model(model_name, ctx, out_len, pretrained=True, model_param=None):
        net = get_model(model_name, pretrained=pretrained)
        with net.name_scope():
            net.output = nn.Dense(out_len)
        if model_param is None:
            net.output.initialize(init.Xavier(), ctx=ctx)
        else:
            net.load_parameters(model_param)
        net.collect_params().reset_ctx(ctx)
        net.hybridize()
        return net

