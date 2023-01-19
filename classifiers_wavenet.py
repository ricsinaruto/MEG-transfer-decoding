from cv2 import COVAR_SCALE
import numpy as np
import os
import torch

from torch.nn import Conv1d, Embedding, CrossEntropyLoss, Dropout, ReLU
from scipy.io import loadmat

from wavenets_simple import WavenetSimple, WavenetSimpleSTS, ConvPoolNet
from wavenets_simple import WavenetSimpleSembConcat, WavenetSimpleSembAdd
from wavenets_simple import WavenetSimpleSembMult, WavenetSimpleChetSemb
from wavenets_full import WavenetSimpleSkips
from classifiers_simpleNN import SimpleClassifier, SimpleClassifierTimeEncoding, ClassifierModule, accuracy


class WavenetClassifier(SimpleClassifier):
    '''
    This class adds a classifier on top of the normal wavenet.
    '''
    def loaded(self, args):
        super(WavenetClassifier, self).loaded(args)
        self.wavenet.loaded(args)

    def kernel_network_FIR(self):
        self.wavenet.kernel_network_FIR()

    def analyse_kernels(self):
        self.wavenet.analyse_kernels()

    def kernelPFI(self, data, sid=None):
        return self.wavenet.kernelPFI(data, sid)

    def build_model(self, args):
        if args.wavenet_class:
            self.wavenet = args.wavenet_class(args)
        else:
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


class WavenetContClass(WavenetClassifier):
    def __init__(self, args):
        super(WavenetContClass, self).__init__(args)

        # provide class weights to deal with unbalanced
        weights = torch.ones(args.num_classes)
        weights[-1] = self.args.epoch_ratio
        self.criterion_class = CrossEntropyLoss(
            weight=weights,
            label_smoothing=args.label_smoothing).cuda()

    def loss(self, x, i=0, sid=None, train=True, criterion=None):
        inputs = x[:, :self.args.num_channels, :]
        targets = x[:, -1, 0].long()
        out_pred, out_class = self.forward(inputs, sid)

        inds = range(targets.shape[0])
        if self.args.no_nonepoch:
            inds = targets < 118

        # compute loss
        loss = self.criterion_class(out_class[inds], targets[inds])

        # compute accuracy
        acc_raw = accuracy(out_class, targets).float()
        acc = torch.mean(acc_raw)

        # look at epoch accuracy only
        inds = targets < 118
        acc_ep = accuracy(out_class[inds], targets[inds])
        acc_ep = torch.mean(acc_ep.float())

        # look at accuracy where full epoch is visible to model
        tresh = int(self.args.sr_data * self.args.decode_peak)
        peak_times = x[:, -2, 0].long()
        inds = (tresh + 1 < peak_times) & (peak_times < tresh + 25) & inds

        acc_full = accuracy(out_class[inds], targets[inds])
        acc_full = torch.mean(acc_full.float())

        # assemble dictionary of losses
        losses = {'trainloss/optloss/Training loss: ': loss,
                  'trainloss/Train accuracy: ': acc,
                  'valloss/Validation loss: ': loss,
                  'valloss/valcriterion/Validation accuracy: ': acc,
                  'valloss/saveloss/none': 1-acc,
                  'trainloss/Train accuracy epoch: ': acc_ep,
                  'valloss/Validation accuracy epoch: ': acc_ep,
                  'trainloss/Train accuracy full-epoch: ': acc_full,
                  'valloss/Validation accuracy full-epoch: ': acc_full}

        return losses, (out_class, out_pred), acc_raw


class WavenetContTimingClass(WavenetContClass):
    def __init__(self, args):
        args.num_channels += 1
        super(WavenetContClass, self).__init__(args)


class WavenetContPeakClass(WavenetContClass):
    def __init__(self, args):
        super(WavenetContPeakClass, self).__init__(args)
        self.criterion_peak = CrossEntropyLoss(
            label_smoothing=args.label_smoothing).cuda()

    def loss(self, x, i=0, sid=None, train=True, crit=None):
        losses, outs, acc_raw = super(WavenetContPeakClass, self).loss(
            x, i, sid, train, crit)
        out = outs[1]

        # targets for peak classification
        targets = x[:, -2, 0].long()
        inds = targets != -1
        loss = self.criterion_peak(out[inds], targets[inds])

        # add the 2 losses together
        losses['trainloss/optloss/Training loss: '] += loss
        losses['valloss/Validation loss: '] += loss

        # accuracy for peak classification
        acc = accuracy(out[inds], targets[inds])

        # look at examples where peak prediction is correct
        out_class = outs[0][inds]
        targets = x[inds, -1, 0].long()

        # calculate accuracy at good peak prediction
        inds = acc == 1
        acc_at_peak = accuracy(out_class[inds], targets[inds])
        acc_at_peak = torch.mean(acc_at_peak.float())

        acc = torch.mean(acc.float())

        losses['trainloss/Train accuracy peak: '] = acc
        losses['valloss/Validation accuracy peak: '] = acc
        losses['trainloss/Train acc class peak: '] = acc_at_peak
        losses['valloss/Validation acc class peak: '] = acc_at_peak

        return losses, out, acc_raw

    def build_model(self, args):
        super(WavenetContPeakClass, self).build_model(args)

        num_classes = args.num_classes
        args.num_classes = args.sample_rate
        self.peak_classifier = ClassifierModule(args, self.class_dim)
        args.num_classes = num_classes

    def forward(self, x, sid=None):
        '''
        Run wavenet on input then feed the output into the classifier.
        '''
        _, x = self.wavenet(x, sid)
        x = x[:, :, ::self.args.rf].reshape(x.shape[0], -1)
        xc = self.classifier(x)

        # peak classification
        xp = self.peak_classifier(x)

        return xp, xc


class ConvPoolClassifier(WavenetClassifier):
    def output_size(self, args):
        time_d = args.sample_rate
        ks = args.kernel_size

        for i in args.dilations:
            time_d -= ks-1
            time_d = int(time_d/2)

        return time_d

    def build_model(self, args):
        self.wavenet = args.wavenet_class(args)

        self.class_dim = self.wavenet.ch * self.output_size(args)
        self.classifier = ClassifierModule(args, self.class_dim)

    def forward(self, x, sid=None):
        '''
        Run wavenet on input then feed the output into the classifier.
        '''
        output, x = self.wavenet(x, sid)
        x = self.classifier(x.reshape(x.shape[0], -1))

        return output, x


class ConvPoolContClass(ConvPoolClassifier, WavenetContClass):
    pass


class ConvPoolContTimingClass(ConvPoolClassifier, WavenetContTimingClass):
    pass


class ConvPoolContPeakClass(ConvPoolClassifier, WavenetContPeakClass):
    def build_model(self, args):
        ConvPoolClassifier.build_model(self, args)

        num_classes = args.num_classes
        args.num_classes = args.sample_rate
        self.peak_classifier = ClassifierModule(args, self.class_dim)
        args.num_classes = num_classes

    def forward(self, x, sid=None):
        '''
        Run wavenet on input then feed the output into the classifier.
        '''
        output, x = self.wavenet(x, sid)
        x = x.reshape(x.shape[0], -1)
        xc = self.classifier(x)

        # peak classification
        xp = self.peak_classifier(x)

        return xp, xc


class ConvPoolTimeEncoding(ConvPoolClassifier, SimpleClassifierTimeEncoding):
    def build_model(self, args):
        chn = args.num_channels
        args.num_channels += args.pos_enc_d
        self.wavenet = ConvPoolNet(args)
        args.num_channels = chn

        self.class_dim = self.wavenet.ch * self.output_size(args)
        self.classifier = ClassifierModule(args, self.class_dim)

    def forward(self, x, sid=None):
        '''
        Run a dimension reduction over the channels then run the classifier.
        '''
        x = self.embed(x)
        return ConvPoolClassifier.forward(self, x, sid)


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
    def set_sub_dict(self):
        # this dictionary is needed because
        # subject embeddings and subjects have a different ordering
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
        super(WavenetClassifierSemb, self).__init__(args)
        self.set_sub_dict()

    def loaded(self, args):
        super(WavenetClassifierSemb, self).loaded(args)
        self.set_sub_dict()

        path = False

        # change embedding to an already trained one
        if 'trained_semb' in args.result_dir:
            path = os.path.join(args.load_model, '..', 'sub_emb.mat')
            
        if 'true_semb' in args.result_dir:
            path = os.path.join(args.result_dir, '..', '..',
                                'finetune_0', 'train1.0', 'sub_emb.mat')

        if path:
            semb = torch.tensor(loadmat(path)['X']).cuda()
            self.wavenet.subject_emb.weight = torch.nn.Parameter(semb)

        # freeze model parameters except subject embeddings
        # if 'freeze' in result_dir
        if 'freeze' in args.result_dir:
            for name, param in self.named_parameters():
                #print(name)
                if 'subject_emb' not in name:
                    param.requires_grad = False

        self.criterion_class_nored = CrossEntropyLoss(reduction='none').cuda()

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
        ind = int(self.args.result_dir.split('_')[-1].split('/')[0])
        ind = self.sub_dict[ind]

        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def get_sid_exc(self, sid):
        '''
        Get subject embedding of untrained subject
        '''
        ind = int(self.args.result_dir.split('_')[-1].split('/')[0])
        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def get_sid_best(self, sid):
        ind = 8
        sid = torch.LongTensor([ind]).repeat(*list(sid.shape)).cuda()
        return sid

    def ensemble_forward(self, x, sid):
        outputs = []
        for i in range(15):
            subid = torch.LongTensor([i]).repeat(*list(sid.shape)).cuda()
            _, out_class = super(WavenetClassifierSemb, self).forward(x, subid)

            outputs.append(out_class.detach())

        outputs = torch.stack(outputs)
        # apply soft max to last dimension of outputs
        outputs = torch.nn.functional.softmax(outputs, dim=-1)

        # average over subject embeddings
        outputs = torch.mean(outputs, dim=0)

        print(torch.argmax(outputs[0], dim=-1))

        return None, outputs

    def forward(self, x, sid=None):
        if not self.args.keep_sid:
            if 'sub' in self.args.result_dir:
                sid = self.get_sid(sid)
            if 'exc' in self.args.result_dir:
                sid = self.get_sid_exc(sid)
            if 'best' in self.args.result_dir:
                sid = self.get_sid_best(sid)
            if 'ensemble' in self.args.result_dir:
                return self.ensemble_forward(x, sid)

        return super(WavenetClassifierSemb, self).forward(x, sid)


class WavenetClassifierSembCov(WavenetClassifierSemb):
    def __init__(self, args):
        super(WavenetClassifierSembCov, self).__init__(args)

        # going from sid to correct subject number
        self.inv_sub_dict = {v: k for k, v in self.sub_dict.items()}

        covs = []
        for i in range(args.subjects):
            path = os.path.join(args.data_path, '..', 'cont',
                                'subj{}_cov.npy'.format(i))
            covs.append(torch.tensor(np.load(path)))

        self.covs = torch.stack(covs).float().cuda()
        self.covs = self.covs[:, ::20]

        # permute according to inv_sub_dict
        self.covs = self.covs[[self.inv_sub_dict[i] for i in range(args.subjects)]]

        # linear layer going from covariance to lower dim embedding
        self.sub_emb = torch.nn.Linear(self.covs.shape[1], args.embedding_dim)

    def build_model(self, args):
        self.wavenet = args.wavenet_class(args)

        self.wavenet.inp_ch = args.num_channels + args.embedding_dim
        self.wavenet.build_model(args)
        
        self.class_dim = self.wavenet.ch * int(args.sample_rate/args.rf)
        self.classifier = ClassifierModule(args, self.class_dim)

        self.cov_drop = Dropout(0.995)
        self.cov_activation = ReLU()

    def forward(self, x, sid=None):
        # get subject-specific covariance
        covs = self.covs[sid]

        # project to embedding space
        embs = self.sub_emb(covs)

        # reshape to same shape as x and concatenate
        embs = embs.repeat(x.shape[2], 1, 1).permute(1, 2, 0)
        x = torch.cat([x, embs], dim=1)

        return super(WavenetClassifierSembCov, self).forward(x, sid)


class WavenetClassifierSembCovD(WavenetClassifierSembCov):
    def set_sub_dict(self):
        self.sub_dict = {0: 3,
                         1: 4,
                         2: 5,
                         3: 6,
                         4: 0,
                         5: 1,
                         6: 2,
                         7: 7,
                         8: 8}

    def __init__(self, args):
        super(WavenetClassifierSembCov, self).__init__(args)

        # going from sid to correct subject number
        self.inv_sub_dict = self.sub_dict

        covs = []
        for i in range(args.subjects):
            path = os.path.join(args.data_path,
                                'cov{}.npy'.format(i))
            cov = np.load(path)

            # take triu of cov
            cov = np.triu(cov).reshape(-1)
            cov = cov[cov != 0]
            covs.append(torch.tensor(cov))

        self.covs = torch.stack(covs).float().cuda()

        # permute according to inv_sub_dict
        self.covs = self.covs[[self.inv_sub_dict[i] for i in range(args.subjects)]]

        # linear layer going from covariance to lower dim embedding
        self.sub_emb = torch.nn.Linear(self.covs.shape[1], args.embedding_dim)


class WavenetClassifierSembCovD7(WavenetClassifierSembCovD):
    def set_sub_dict(self):
        self.sub_dict = {0: 3,
                         1: 4,
                         2: 5,
                         3: 6,
                         4: 0,
                         5: 1,
                         6: 2}

    def loaded(self, args):
        super(WavenetClassifierSembCovD7, self).loaded(args)
        self.set_sub_dict()

        covs = []
        path = os.path.join(args.data_path, '..', 'cov4.npy')
        cov = np.load(path)

        # take triu of cov
        cov = np.triu(cov).reshape(-1)
        cov = cov[cov != 0]
        covs.append(torch.tensor(cov))

        self.covs = torch.stack(covs).float().cuda()


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
