import numpy as np
import os
import torch

from torch.nn import Conv1d, Embedding

from wavenets_simple import WavenetSimple, WavenetSimpleSTS
from wavenets_simple import WavenetSimpleSembConcat, WavenetSimpleSembAdd
from wavenets_simple import WavenetSimpleSembMult, WavenetSimpleChetSemb
from wavenets_full import WavenetSimpleSkips
from classifiers_simpleNN import SimpleClassifier, ClassifierModule, accuracy


class WavenetClassifier(SimpleClassifier):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''
    def loaded(self, args):
        super(WavenetClassifier, self).loaded(args)
        self.wavenet.loaded(args)

    def kernel_network_FIR(self):
        self.wavenet.kernel_network_FIR()

    def build_model(self, args):
        self.wavenet = WavenetSimple(args)

        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
        self.classifier = ClassifierModule(args, self.class_dim)

    def forward(self, x, sid=None):
        '''
        Run wavenet on input then feed the output into the classifier.
        '''
        output, x = self.wavenet(x, sid)
        x = x[:, :, ::self.args.rf].reshape(x.shape[0], -1)
        x = self.classifier(x)

        return output, x


class WavenetClassifierClamped(WavenetClassifier):
    def forward(self, x, sid=None):
        # clamp weights before running forward pass
        for layer in self.wavenet.cnn_layers:
            layer.weight = self.clamp(layer.weight)
            layer.bias = self.clamp(layer.bias)

        for layer in self.classifier.layers:
            layer.weight = self.clamp(layer.weight)
            layer.bias = self.clamp(layer.bias)

        return super(WavenetClassifierClamped, self).forward(x, sid)

    def clamp(self, tensor):
        tensor = torch.clamp(tensor, -0.07, 0.07)
        return torch.nn.Parameter(tensor)


class WavenetSkipsClassifier(WavenetClassifier):
    '''
    Adds a classifier on top of the WavenetSimpleSkips model.
    '''
    def build_model(self, args):
        super(WavenetSkipsClassifier, self).build_model(args)
        self.wavenet = WavenetSimpleSkips(args)


class WavenetClassifierSTS(WavenetClassifier):
    '''
    Adds a classifier on top of the WavenetSimpleSTS model.
    '''
    def build_model(self, args):
        super(WavenetClassifierSTS, self).build_model(args)
        self.wavenet = WavenetSimpleSTS(args)


class WavenetClassifierRed(WavenetClassifier):
    '''
    Wavenet Classifier with extra dimensionality reduction.
    '''
    def build_model(self, args):
        self.wavenet = WavenetSimple(args)
        self.class_dim = args.dim_red * int(args.sample_rate/args.rf)

        self.classifier = ClassifierModule(args, self.class_dim)
        self.spatial_conv = Conv1d(
            self.wavenet.ch, args.dim_red, kernel_size=1, groups=1)

    def forward(self, x, sid):
        output, x = self.wavenet(x)
        x = x[:, :, ::self.args.rf]

        # reduce channel dimension before feeding into the classifier
        x = self.spatial_conv(x)
        x = self.classifier.activation(self.classifier.dropout(x))
        x = self.classifier(x.reshape(x.shape[0], -1))

        return output, x


class WavenetClassifierConcat(WavenetClassifier):
    '''
    Wavenet Classifier with full output from wavenet instead of downsampling.
    '''
    def build_model(self, args):
        self.wavenet = WavenetSimple(args)
        self.class_dim = args.dim_red * (args.sample_rate - args.rf + 1)

        self.classifier = ClassifierModule(args, self.class_dim)
        self.spatial_conv = Conv1d(
            self.wavenet.ch, args.dim_red, kernel_size=1, groups=1)

    def forward(self, x, sid):
        output, x = self.wavenet(x)

        self.spatial_conv(x)
        x = self.classifier.activation(self.classifier.dropout(x))
        x = self.classifier(x.reshape(x.shape[0], -1))

        return output, x


class WavenetClassifierTimewise(WavenetClassifier):
    '''
    Wavenet Classifier with a separate class prediction at each timestep.
    '''
    def build_model(self, args):
        self.wavenet = WavenetSimple(args)
        self.classifier = ClassifierModule(args, self.wavenet.ch)

    def forward(self, x, sid):
        output, x = self.wavenet(x)

        # use the same classifier to a make class prediction at each timestep
        x = x.permute(0, 2, 1)
        x = self.classifier(x)

        # mean class prediction across timesteps
        x = torch.mean(x, dim=1)

        return output, x


class WavenetClassifierTimeloss(WavenetClassifierTimewise):
    '''
    WavenetClassifier with a separate class prediction at each timestep.
    The class predictions are not averaged across timesteps.
    '''
    def loaded(self, args):
        super(WavenetClassifierTimeloss, self).loaded(args)
        self.timeacc = None

    def forward(self, x, sid):
        output, x = self.wavenet(x)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)

        return output, x

    def loss(self, x, i=0, sid=None, train=True, criterion=None):
        '''
        Run the model in forward mode and compute loss for this batch.
        '''
        inputs = x[:, :self.args.num_channels, :]
        out_pred, out_class = self.forward(inputs, sid)

        targets = x[:, -1, :out_class.shape[1]].long()
        acc = accuracy(out_class, targets).float()

        # save accuracy for over all timesteps
        self.save_timeacc(acc.cpu(), i)
        if criterion is None:
            acc = torch.mean(acc)

        targets = targets.reshape(-1)
        out_class = out_class.reshape(-1, self.args.num_classes)

        loss = self.criterion_class_nored(out_class, targets)
        if not train:
            loss = torch.quantile(loss, 0.4)
        loss = torch.mean(loss)

        return loss, out_pred, None, acc

    def save_timeacc(self, acc, i):
        '''
        Save accuracies across timesteps.
        '''
        if self.timeacc:
            self.timeacc = torch.cat((self.timeacc, acc), dim=0)
        else:
            self.timeacc = acc
        print(self.timeacc.shape)

        if i == 59:
            timeacc = torch.mean(self.timeacc, dim=0).flatten().tolist()
            timeacc = [str(i) for i in timeacc]
            path = os.path.join(self.args.result_dir, 'time_acc.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(timeacc))


class WavenetClassifierSemb(WavenetClassifier):
    '''
    Wavenet Classifier for multi-subject data using subject embeddings.
    '''
    def build_model(self, args):
        self.wavenet = args.wavenet_class(args)

        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
        self.classifier = ClassifierModule(args, self.class_dim)

    def save_embeddings(self):
        self.wavenet.save_embeddings()

    def loaded_(self, args):
        super(WavenetClassifierSemb, self).loaded(args)

        # set embeddings to 0 (not normally used)
        self.wavenet.subject_emb = Embedding(args.subjects, args.embedding_dim)

        shape = tuple(list(self.wavenet.subject_emb.weight.shape))
        zeros = torch.zeros(shape).cuda()
        self.wavenet.subject_emb.weight = torch.nn.Parameter(zeros)

    def get_sid(self, sid):
        '''
        Get subject id based on result directory name.
        '''
        ind = int(self.args.result_dir.split('_')[-1])
        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()

        return sid

    def forward_(self, x, sid=None):
        '''
        Use the subject embedding of a subject based on result directory name.
        Not normally used.
        '''
        sid = self.get_sid(sid)
        return super(WavenetClassifierSemb, self).forward(x, sid=sid)


class WavenetClassifierSembChet(WavenetClassifierSemb):
    '''
    Multi-subject Wavenet classifier using WavenetSimpleChetSemb.
    '''
    def build_model(self, args):
        self.wavenet = WavenetSimpleChetSemb(args)

        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
        self.classifier = ClassifierModule(args, self.class_dim)


class WavenetClassifierSembAdd(WavenetClassifierSemb):
    '''
    Multi-subject Wavenet classifier using WavenetSimpleSembAdd.
    '''
    def build_model(self, args):
        self.wavenet = WavenetSimpleSembAdd(args)
        WavenetClassifier.build_model(self, args)


class WavenetClassifierSembMult(WavenetClassifierSemb):
    '''
    Multi-subject Wavenet classifier using WavenetSimpleSembMult.
    '''
    def build_model(self, args):
        self.wavenet = WavenetSimpleSembMult(args)
        WavenetClassifier.build_model(self, args)


class WavenetClassPred(WavenetClassifier):
    '''
    Wavenet Classifier pretrained for next-timestep prediction
    '''
    def __init__(self, args):
        super(WavenetClassPred, self).__init__(args)
        self.alpha = args.norm_alpha

    def forward(self, x, sid=None):
        # use only the bottom few layers of Wavenet
        output, x = self.wavenet.forward4(x, sid)
        x = x[:, :, ::self.args.rf].reshape(x.shape[0], -1)
        x = self.classifier(x)

        return output, x

    def get_weights(self, grad=False):
        '''
        Get weights of the full wavenet+classifier modules.
        '''
        # use different get weight function, whether we need gradient
        get_weight = self.wavenet.get_weight_nograd
        if grad:
            get_weight = self.wavenet.get_weight

        # classifier weights
        weights = [get_weight(layer) for layer in self.classifier.layers]

        # wavenet weights
        weights.extend(self.wavenet.get_weights(grad))

        return weights

    def loaded(self, args):
        super(WavenetClassPred, self).loaded(args)
        self.alpha = args.norm_alpha
        self.initial_weights = self.get_weights()

        if args.init_model:
            self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
            self.classifier = ClassifierModule(args, self.class_dim)

            # extract initial weights (trained for forecasting)
            self.initial_weights = self.wavenet.get_weights()

        if args.fixed_wavenet:
            self.fix_weights()

    def fix_weights(self):
        # fix all wavenet weights
        for parameter in self.wavenet.parameters():
            parameter.requires_grad = False

    def loss_reg(self):
        '''
        Compute L2 loss between new and initial weights.
        '''
        new_weights = self.get_weights(grad=True)
        if self.args.init_model:
            new_weights = self.wavenet.get_weights(grad=True)

        loss = []
        for nw, ow in zip(new_weights, self.initial_weights):
            loss.append(torch.linalg.norm(nw - ow))

        return torch.sum(torch.stack(loss))

    def loss(self, x, i=0, sid=None, train=True, pred=False, criterion=None):
        '''
        Run the model in forward mode and compute loss for this batch.
        '''
        losses = {}
        if self.args.pred or pred:
            # run the model to forecast
            loss, _, _ = self.wavenet.loss(x, i, sid, train)
            full_loss = loss['trainloss/optloss/Training loss: ']
            losses['valloss/saveloss/none'] = full_loss
        else:
            # run the model to classify
            loss, out_pred, _ = super(WavenetClassPred, self).loss(x, sid=sid, train=train, criterion=criterion)
            accuracy = loss['trainloss/Train accuracy: ']

            loss_reg = 0
            full_loss = loss['trainloss/optloss/Training loss: ']
            if self.alpha > 0.0:
                # add regularization based on initial weights
                loss_reg = self.alpha * self.loss_reg()
                full_loss += loss_reg
                losses['trainloss/Training regularization: '] = loss_reg
                losses['valloss/Validation regularization: '] = loss_reg

            losses['trainloss/Train accuracy: '] = accuracy
            losses['valloss/valcriterion/Validation accuracy: '] = accuracy
            losses['valloss/saveloss/none'] = loss['valloss/saveloss/none']

        losses['trainloss/optloss/Training loss: '] = full_loss
        losses['valloss/Validation loss: '] = full_loss

        return losses, None, None

    def compare_layers(self):
        '''
        Compare newly learned and initial weights of individual layers.
        '''
        old_model = torch.load(self.args.compare_model)

        diffs = []
        d = self.layer_diff(self.wavenet.first_conv,
                            old_model.wavenet.first_conv)
        diffs.append(d)
        for i in range(int(np.log(self.args.rf) / np.log(2))):
            d = self.layer_diff(self.wavenet.cnn_layers[i],
                                old_model.wavenet.cnn_layers[i])
            diffs.append(d)

        # add classification weights
        for i in range(len(self.classifier.layers)):
            d = self.layer_diff(self.classifier.layers[i],
                                old_model.classifier.layers[i])
            diffs.append(d)

        path = os.path.join(self.args.result_dir, 'layer_diff.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(diffs))

    def layer_diff(self, new, old):
        # MSE between new and old weights
        with torch.no_grad():
            diff = torch.nn.functional.mse_loss(new.weight, old.weight)
            return str(diff.item())


class WavenetClassPredL2(WavenetClassPred):
    '''
    WavenetClassPred with L2 loss on wavenet weights.
    '''
    def loss_reg(self):
        new_weights = self.wavenet.get_weights(grad=True)

        loss = []
        for nw in new_weights:
            loss.append(torch.linalg.norm(nw))

        return torch.sum(torch.stack(loss))


class WavenetClassPredSemb(WavenetClassPred, WavenetClassifierSemb):
    '''
    This class makes WavenetClassPred to work with subject embeddings.
    '''
    def set_sub_dict(self):
        # this dictionary is needed because
        # continuous and epoched subjects have a different ordering
        self.sub_dict = {0: 10,
                         1: 7,
                         2: 3,
                         3: 11,
                         4: 8,
                         5: 4,
                         6: 12,
                         7: 9,
                         8: 5,
                         9: 13,
                         10: 1,
                         11: 14,
                         12: 2,
                         13: 6,
                         14: 0}

    def __init__(self, args):
        super(WavenetClassPredSemb, self).__init__(args)
        self.set_sub_dict()

    def loaded(self, args):
        super(WavenetClassPredSemb, self).loaded(args)
        self.set_sub_dict()

    def get_sid(self, sid):
        '''
        Get subject id based on result directory name.
        '''
        ind = int(self.args.result_dir.split('_')[-1].split('/')[0])
        ind = self.sub_dict[ind]

        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def forward(self, x, sid=None):
        if 'sub' in self.args.result_dir:
            sid = self.get_sid(sid)

        return super(WavenetClassPredSemb, self).forward(x, sid)


class WavenetCombined(WavenetClassPred):
    '''
    This model is trained simultaneously for prediction and classification.
    '''
    def loss(self, x, i=0, sid=None, train=True):
        losses = {}

        # get both the prediction and classification loss
        class_loss, _, _ = super(WavenetCombined, self).loss(
            x['epoched'], train=train, sid=sid)
        pred_loss, _, _ = super(WavenetCombined, self).loss(
            x['cont'], train=train, pred=True, sid=sid)

        accuracy = class_loss['trainloss/Train accuracy: ']
        cls_loss_full = class_loss['trainloss/optloss/Training loss: ']
        pred_loss_full = pred_loss['trainloss/optloss/Training loss: ']
        full_loss = pred_loss_full * self.args.pred_multiplier + cls_loss_full

        losses['trainloss/Train accuracy: '] = accuracy
        losses['valloss/valcriterion/Validation accuracy: '] = accuracy
        losses['valloss/saveloss/none'] = class_loss['valloss/saveloss/none']

        losses['trainloss/optloss/Training loss: '] = full_loss
        losses['valloss/Validation loss: '] = full_loss

        return losses, None, None


class WavenetCombinedSemb(WavenetCombined, WavenetClassifierSemb):
    '''
    WavenetCombined for multi-subject data with subject embeddings.
    '''
    pass


class WavenetSTSCombined(WavenetCombined):
    '''
    WavenetCombined using the WavenetSimpleSTS model.
    '''
    def build_model(self, args):
        super(WavenetSTSCombined, self).build_model(args)
        self.wavenet = WavenetSimpleSTS(args)


class WavenetSTSClassPred(WavenetClassPred):
    '''
    WavenetClassPred using the WavenetSimpleSTS model.
    '''
    def build_model(self, args):
        super(WavenetSTSClassPred, self).build_model(args)
        self.wavenet = WavenetSimpleSTS(args)
