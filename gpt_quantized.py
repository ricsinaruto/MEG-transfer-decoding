from torch.nn import CrossEntropyLoss, Embedding, Linear, ModuleList, Module
from torch.nn.parameter import Parameter
import torch
import os
import numpy as np
import gc
from scipy.io import savemat

from wavenets_full import WavenetFullChannelMixMixin, sample, accuracy
from wavenets_simple import topk_accuracy

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from transformers.models.reformer.modeling_reformer import ReformerModel


class Embeddings(Module):
    def __init__(self, args, num_channels):
        super().__init__()
        self.args = args
        self.num_channels = num_channels

        # embeddings for various conditioning
        self.cond_emb = Embedding(args.num_classes, args.class_emb)

        # quantization embedding
        self.quant_emb = Embedding(args.quant_levels, args.quant_emb)

        # channel embeddings
        self.ch_emb = Embedding(args.num_channels, args.channel_emb)

        # subject embeddings
        self.sub_emb = Embedding(args.subjects, args.embedding_dim)

        # initialize weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        '''
        Initialize weights of embedding layers

        Args:
            module (torch.nn.Module): module to initialize
        '''
        if isinstance(module, Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def add_embeds(self, x, ch_emb, cond_emb, semb):
        # add up embeddings
        x = x + ch_emb + cond_emb + semb
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # B*C x T x E
        return x

    def forward(self, x, ids, cond=None, sid=None):
        # apply quantization embedding
        x = self.quant_emb(x)  # B x C x T x E_q
        timesteps = x.shape[2]
        channels = x.shape[1]

        cond_emb = 0
        if self.args.class_emb > 0 and cond is not None:
            cond_emb = self.cond_emb(cond)  # B x 1 x T x E_c

            # set elements of cond to 0 where id is 0
            inds = cond.unsqueeze(-1) > 0
            cond_emb *= inds.to(cond_emb.dtype)
            cond_emb = cond_emb.expand(-1, channels, -1, -1)  # B x C x T x E_c

        # check if this is a group model
        semb = 0
        if self.args.embedding_dim > 0 and sid is not None:
            # project to embedding space
            semb = self.sub_emb(sid)
            semb = semb.expand(-1, channels, -1, -1)  # B x C x T x E_s

        ch_ids = torch.arange(self.args.num_channels).to(x.device)

        # get channel embedding: C x E_ch
        ch_emb = self.ch_emb(ch_ids[ids])
        # repeat across batch and time:  B x T x C x E_ch
        ch_emb = ch_emb.expand(1, timesteps, -1, -1)
        ch_emb = ch_emb.permute(0, 2, 1, 3)  # B x C x T x E_ch

        return self.add_embeds(x, ch_emb, cond_emb, semb)


class ChnIndependentHead(Module):
    def __init__(self, args, num_channels, out_times):
        super().__init__()

        self.out_times = out_times
        self.num_channels = num_channels
        # output head
        self.head = Linear(args.gpt2_config.n_embd,
                           args.quant_levels,
                           bias=False)

        # initialize head weights
        self.head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        x = self.head(x)  # B*C x T x Q
        return x[:, -self.out_times:, :]


class GPT2MEG(WavenetFullChannelMixMixin):
    def __init__(self, config, args=None):
        super().__init__()
        if args is None:
            args = config

        self.args = args
        self.use_cache = False

        args.quant_levels = args.gpt2_config.vocab_size
        self.quant_levels = args.quant_levels
        self.out_times = args.example_shift
        self.save_preds = False

        # channel dim for model
        self.num_channels = args.num_channels

        # check if args has model_chn_dim and set self.num_channels
        if hasattr(args, 'model_chn_dim'):
            self.num_channels = args.model_chn_dim

        self.build_model(args)
        self.embeddings.out_times = self.out_times
        self.criterion = CrossEntropyLoss(reduction='none').to(args.device)
        self.mse_loss = torch.nn.MSELoss().to(args.device)

        self.new_names = ['head', 'cond_emb', 'quant_emb', 'ch_emb']

    def build_model(self, args):
        self.gpt2 = GPT2Model(args.gpt2_config)
        self.embeddings = Embeddings(args, self.num_channels)
        self.head = ChnIndependentHead(args, self.num_channels, self.out_times)

    def set_shift(self, shift):
        self.out_times = shift

    def loaded(self, args):
        self.args = args
        self.out_times = args.example_shift

        # check if model has num_channels attribute
        if not hasattr(self, 'num_channels'):
            self.num_channels = args.num_channels
            if hasattr(args, 'model_chn_dim'):
                self.num_channels = args.model_chn_dim

        if not hasattr(self, 'use_cache'):
            self.use_cache = False

    def generate(self, train_data):
        self.out_times = 0
        self.use_cache = False
        self.args.rf = self.args.sample_rate
        return super().generate(train_data)

    def reshape_output(self, x):
        return x.reshape(-1, self.num_channels, x.shape[1], x.shape[2])

    def forward(self, data, chid=None):
        x = data['inputs']  # B x C x T
        cond = data.get('condition', None)  # conditions
        sid = data.get('sid', None)  # subject ids

        # get channel ids
        if chid is None:
            chid = data.get('ch_ids', np.arange(self.args.num_channels))

        # embed inputs
        x = self.embeddings(x, chid, cond, sid)

        if 'past_key_values' in data:
            outputs = self.gpt2(inputs_embeds=x,
                                past_key_values=data['past_key_values'],
                                use_cache=True)
        else:
            outputs = self.gpt2(inputs_embeds=x)
            outputs.past_key_values = None

        x = self.reshape_output(self.head(outputs[0]))
        if outputs.past_key_values is None:
            return x

        return (x, outputs.past_key_values)


def create_parameter(shape):
    # Create a random tensor with the given shape.
    tensor = torch.normal(size=shape,
                          dtype=torch.float32,
                          requires_grad=True,
                          device='cuda',
                          mean=0.0,
                          std=0.02)
    # Wrap the tensor in a Parameter to enable autograd.
    return Parameter(tensor)


class ChannelWiseHead(Module):
    def __init__(self, args, num_channels, out_times, hidden=None, head=None):
        super().__init__()

        self.out_times = out_times
        self.num_channels = num_channels

        hidden = 2 * args.gpt2_config.hidden_size if hidden is None else hidden
        shape = (args.num_channels, hidden, args.quant_levels)

        # output heads
        self.head = create_parameter(shape) if head is None else head

    def generate(self, x):
        x = x[:, -1:, :].reshape(1, 1, 1, x.shape[2])

        if self.chid >= self.num_channels:
            return None
        return x @ self.choose_head()[self.chid]

    def choose_head(self):
        if isinstance(self.head, ModuleList):
            return torch.stack([emb.weight.T for emb in self.head])
        return self.head

    def forward(self, x):
        if getattr(self, 'chid', None) is not None:
            return self.generate(x)

        x = x.reshape(x.shape[0], -1, self.num_channels + 1, x.shape[2])

        # remove the separator token
        x = x[:, -self.out_times:, :-1, :]  # B x T x C x E
        b, t, c, e = x.shape

        heads = self.choose_head()

        # apply per-channel head
        x = x.permute(2, 0, 1, 3).reshape(c, b*t, e)
        x = torch.bmm(x, heads)  # C x B*T x Q
        x = x.reshape(c, b, t, -1)  # C x B x T x Q

        return x.permute(1, 0, 2, 3)  # B x C x T x Q


class ListEmbedding(Module):
    def __init__(self, args):
        super().__init__()

        ql, qe, ch = args.quant_levels, args.quant_emb, args.num_channels
        self.emb = ModuleList(Embedding(ql, qe) for _ in range(ch))

    def forward(self, x):
        if getattr(self, 'chid', None) is not None:
            return self.emb[self.chid](x)

        out = []
        for i in range(x.shape[1]):
            out.append(self.emb[i](x[:, i:i+1, :]))
        return torch.cat(out, dim=1)


class EmbeddingsFlat(Embeddings):
    def __init__(self, args, num_channels):
        super().__init__(args, num_channels)

        # create quantization embedding
        self.quant_emb = ListEmbedding(args)

        # create separator embedding
        self.sep_emb = Embedding(1, args.quant_emb)

        # Create timestep embeddings
        self.ts_emb = Embedding(args.sample_rate, args.ts_emb)

        self.apply(self.init_weights)

    def set_chid(self, chid):
        self.quant_emb.chid = chid

    def add_embeds(self, x, ch_emb, cond_emb, semb):
        b, c, t, e = x.shape

        tid = [0, t]
        if getattr(self, 'tid', None) is not None:
            tid = [self.tid, self.tid + 1]

        # get timestep embedding: T x E_ts
        ts_emb = self.ts_emb(torch.arange(tid[0], tid[1]).to(x.device))
        # repeat across batch and channels
        ts_emb = ts_emb.expand(b, c, -1, -1)

        # add up embeddings
        x += cond_emb + ch_emb + ts_emb + semb
        x = x.permute(0, 2, 1, 3).reshape(b, t*c, e)  # B x (T * C) x E

        if b*c*t < 2:
            return x

        # add Q0 to separate timesteps
        # these should come in between the timesteps
        # B x (T1 Q0 T2 Q0 ... TC) x E_q
        sep_ids = torch.zeros(b, 1, dtype=torch.int32).to(x.device)
        sep_emb = self.sep_emb(sep_ids)

        elements = [sep_emb]
        for i in range(0, x.shape[1], self.num_channels):
            elements.append(x[:, i:i+self.num_channels, :])
            elements.append(sep_emb)

        if self.out_times == 1:
            return torch.cat(elements, dim=1)
        return torch.cat(elements[:-1], dim=1)


class GPT2Flat(GPT2MEG):
    '''
    A modification of the TransformerQuantized model that
    flattens the channel x time dimension to a single sequence
    that is fed into the model. To handle long sequences,
    we use a Reformer model instead of GPT2. We aid the model through
    channel and timestep embeddings, and we also use smart padding
    to not attend to current or future timesteps.

    The input to the model is a sequence of quantized values:
    Q11, Q12, Q13, ..., Q0, Q21, Q22, Q23, ..., Q0, Q31, Q32, Q33, ...
    where Qij is the quantized value of the ith timestep of the jth channel.
    Q0 is a special token that is used as separator between timesteps.
    The following embeddings are added to the input sequence:
    - channel embedding: E_ch (one per channel j)
    - timestep embedding: E_ts (one per timestep i)
    - conditional embedding: E_cond (as in TransformerQuantized)

    The model is trained to predict the next value in the sequence.
    When predicting timestep i, the model is only allowed to attend to
    timesteps 1 to i-1, so in addition to future sequence elements,
    we have to mask past elements of the sequence that have timestep i.
    '''
    def build_model(self, args):
        self.gpt2 = ReformerModel(args.gpt2_config)
        self.embeddings = EmbeddingsFlat(args, self.num_channels)
        self.head = ChannelWiseHead(args, self.num_channels, self.out_times)

        self.gpt2.wte = None

    def loaded(self, args):
        super().loaded(args)
        self.embeddings.out_times = self.out_times

    def set_tid(self, tid):
        self.embeddings.tid = tid

    def set_chid_head(self, chid):
        self.head.chid = chid

    def set_chid_embs(self, chid):
        self.embeddings.set_chid(chid)

    def reshape_output(self, x):
        return x

    def forward(self, data, chid=None):
        data['targets'] = data['inputs']
        return super().forward(data, chid=chid)

        # modify targets to match the new sequence length
        '''
        targets = data['inputs'].permute(0, 2, 1)  # B x T x C
        targets = targets.reshape(x.shape[0], -1)  # B x (T * C)

        # include separator ids in targets
        elements = []
        for i in range(0, targets.shape[1], self.num_channels):
            elements.append(targets[:, i:i+self.num_channels])
            elements.append(sep_ids)
        targets = torch.cat(elements, dim=1)

        shape = targets.shape
        targets = targets.reshape(shape[0], -1, self.num_channels + 1)
        data['targets'] = targets.permute(0, 2, 1)  # B x C+1 x T
        '''

    def generate(self, dataset):
        self.eval()
        if hasattr(self.args, 'no_kv_cache'):
            self.generate_nokv(dataset)
            return

        if self.args.amp:
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16):
                self.generate_(dataset)
        else:
            self.generate_(dataset)

    def generate_nokv(self, dataset):
        '''
        Same as self.generate_ but without resuing the key/value cache.
        '''
        self.eval()
        self.use_cache = False

        # out_times should be number of channels
        self.out_times = 1
        self.head.out_times = 1
        channels = self.args.num_channels
        ex_shift = self.args.example_shift
        inp_len = self.args.sample_rate
        gen_len = self.args.generate_length

        xtrain = dataset.x_train_t
        assert xtrain.shape[2] / ex_shift == 2

        data = xtrain[0, :channels, :].clone()
        zeros = torch.zeros((channels, gen_len), dtype=torch.int16)
        data = torch.cat((data, zeros.to(data.device)), dim=1)

        # select half of label timeseries from each batch
        cond = xtrain[:, -2, :xtrain.shape[2]//2].reshape(-1)
        cond = cond[:data.shape[1]]

        # save cond to result_dir for later
        np.save(os.path.join(self.args.result_dir, 'generate_cond.npy'),
                cond.cpu().numpy())
        cond.unsqueeze_(0).unsqueeze_(0)
        data.unsqueeze_(0)

        print(data.shape)
        print(cond.shape)

        for chunk in range(0, data.shape[2]-inp_len):
            end_ind = chunk+inp_len
            inputs = data[:, :, chunk:end_ind]
            cond_ex = cond[:, :, chunk:end_ind]

            for c in range(self.num_channels):
                logits = self.forward({'inputs': inputs, 'condition': cond_ex})

                # select current channel
                logits = logits[:, c, -1, :].detach()
                out = sample(self.args, logits)

                data[:, c, end_ind-1] = out
                inputs[:, c, -1] = out

            
            if chunk < 5:
                print(data[:, :, end_ind-1])


            # print progress in percent
            if chunk % 100 == 0:
                percent = chunk/data.shape[2]*100
                print('Progress: {:.2f}%'.format(percent), flush=True)

        data = data[0, :, :].cpu().numpy()
        name = 'generated.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

        # decode to raw data
        data = dataset.reconstruct(data)
        name = 'generated_decoded.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

    def generate_(self, dataset):
        '''
        Recursively generate with a trained model in various ways.
        '''
        self.out_times = 1
        self.head.out_times = 1
        self.embeddings.out_times = 1
        self.use_cache = True
        channels = self.args.num_channels
        ex_shift = self.args.example_shift
        inp_len = self.args.sample_rate
        gen_shift = self.args.generate_shift
        gen_len = self.args.generate_length

        xtrain = dataset.x_train_t
        assert xtrain.shape[2] / ex_shift == 2

        data = xtrain[0, :channels, :-gen_shift].clone()
        zeros = torch.zeros((channels, gen_len), dtype=torch.int16)
        data = torch.cat((data, zeros.to(data.device)), dim=1)

        # select half of label timeseries from each batch
        cond = xtrain[:, -2, :xtrain.shape[2]//2].reshape(-1)
        cond = cond[:data.shape[1]]

        # save cond to result_dir for later
        np.save(os.path.join(self.args.result_dir, 'generate_cond.npy'),
                cond.cpu().numpy())
        cond.unsqueeze_(0).unsqueeze_(0)
        data.unsqueeze_(0)

        print(data.shape)
        print(cond.shape)

        for chunk in range(0, data.shape[2]-inp_len, gen_shift):
            end_ind = chunk+inp_len-gen_shift
            inputs = data[:, :, chunk:end_ind]
            cond_ex = cond[:, :, chunk:end_ind]

            self.set_chid_head(0)
            skip = False
            past_kv = None

            # Release unused GPU memory and monitor memory usage
            # torch.cuda.empty_cache()
            # print("Memory allocated: ", torch.cuda.memory_allocated())
            # print("Memory cached: ", torch.cuda.memory_reserved())

            for t in range(gen_shift):
                for c in range(self.num_channels+1):
                    if not skip:
                        chid = np.array([c-1]) if c > 0 else None
                        logits, past_kv = self.forward({
                            'inputs': inputs,
                            'condition': cond_ex,
                            'past_key_values': past_kv}, chid=chid)
                        if logits is not None:
                            logits = logits.detach()
                            inputs = sample(self.args, logits)

                    self.set_tid(end_ind+t-chunk)
                    self.set_chid_embs(c)
                    self.set_chid_head(c+1)
                    skip = False

                    cond_ex = cond[:, :, end_ind+t:end_ind+t+1]

                    if c < self.num_channels:
                        data[:, c:c+1, end_ind+t:end_ind+t+1] = inputs
                    elif t < gen_shift - 1:
                        # add separator token
                        sep_ids = torch.zeros(1, 1, dtype=torch.int32)
                        sep_ids = sep_ids.to(data.device)
                        sep_emb = self.embeddings.sep_emb(sep_ids)

                        out = self.gpt2(inputs_embeds=sep_emb,
                                        past_key_values=past_kv,
                                        use_cache=True)
                        past_kv = out.past_key_values

                        self.set_chid_head(0)
                        out = out[0].detach()
                        out = self.reshape_output(self.head(out))
                        inputs = sample(self.args, out)
                        skip = True

                # if chunk < 10:
                #    print(data[:, :, end_ind+t])

            self.set_tid(None)
            self.set_chid_embs(None)

            # print progress in percent
            if chunk % 100 == 0:
                percent = chunk/data.shape[2]*100
                print('Progress: {:.2f}%'.format(percent), flush=True)

        data = data[0, :, :].cpu().numpy()
        name = 'generated.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

        # decode to raw data
        data = dataset.reconstruct(data)
        name = 'generated_decoded.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

        # check reconstruction of real data
        xtrain = xtrain[:, :channels, :xtrain.shape[2]//2]
        xtrain = xtrain.permute(1, 0, 2).reshape(channels, -1)
        recon = dataset.reconstruct(xtrain.cpu().numpy())

        # save reconstruction to result_dir for later
        name = 'reconstruction_xtrain.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': recon})

    def crop_kv(self, kv, ind):
        new_kv = []
        for lyr in kv:
            tup = (lyr[0][:, :, :ind, :], lyr[1][:, :, :ind, :])
            new_kv.append(tup)

        return tuple(new_kv)


class GPT2Flat_fullattention(GPT2Flat):
    def build_model(self, args):
        self.embeddings = EmbeddingsFlat(args, self.num_channels)
        self.gpt2 = GPT2Model(args.gpt2_config)
        self.head = ChannelWiseHead(args,
                                    self.num_channels,
                                    self.out_times,
                                    head=self.embeddings.quant_emb.emb)

        self.gpt2.wte = None

    def loss(self, data, i=0, sid=None, train=True, criterion=None):
        '''
        Recursively calculate the loss of a trained model
        for every channel in the current timestep.
        '''
        if not getattr(self.args, 'recursive_loss', None):
            return self.loss_(data, i, sid, train, criterion)

        sr = self.args.sample_rate
        shift = self.args.generate_shift
        num_chn = self.args.num_channels + 1
        self.out_times = shift
        self.head.out_times = shift
        self.embeddings.out_times = None
        self.use_cache = True

        self.set_tid(None)
        self.set_chid_embs(None)
        self.set_chid_head(None)

        inputs = data['inputs']
        cond = data['condition']
        out_og, past_kv_og = self.forward({'inputs': inputs,
                                           'condition': cond,
                                           'past_key_values': None})
        out_og = out_og.detach()

        dims = inputs[:, :, -shift:].shape
        dims = tuple([d for d in dims] + [out_og.shape[-1]])
        outputs = torch.zeros(dims).to(inputs.device)
        sampled = torch.zeros(inputs[:, :, -shift:].shape).to(inputs.device)
        sampled = sampled.to(torch.int32)

        '''
        # set first channel with out_og
        outputs[:, :1, :] = out_og[:, :1, :]
        sampled[:, :1, :] = sample(self.args, out_og[:, :1, :])

        # only predict one timestep
        self.out_times = 1
        self.head.out_times = 1
        self.embeddings.out_times = 1

        for t in range(shift):
            past_ind = (sr-shift+t) * num_chn + 1
            past_kv = self.crop_kv(past_kv_og, past_ind)

            inputs = sampled[:, :1, t:t+1]
            cond_ex = cond[:, :, sr-shift+t:sr-shift+t+1]
            self.set_tid(sr-shift+t)

            for c in range(self.num_channels-1):
                self.set_chid_embs(c)
                self.set_chid_head(c+1)

                logits, past_kv = self.forward({
                    'inputs': inputs,
                    'condition': cond_ex,
                    'past_key_values': past_kv},
                    chid=np.array([c]))

                logits = logits.detach()
                inputs = sample(self.args, logits)

                outputs[:, c+1:c+2, t:t+1] = logits
                sampled[:, c+1:c+2, t:t+1] = inputs
        '''
        outputs = out_og
        sampled = sample(self.args, out_og)

        # calculate loss
        metrics = self.metrics(outputs, data['inputs'])[:3]
        losses = self.pack_loss(*metrics)

        # compute mse of reconstructed data
        targets = data['inputs'][:, :, -outputs.shape[-2]:]
        targets = targets.reshape(self.args.num_channels, -1)
        targets = self.ds.reconstruct(targets.cpu().numpy())

        sampled = sampled.reshape(self.args.num_channels, -1)
        sampled = self.ds.reconstruct(sampled.cpu().numpy())

        # calculate mse
        mse = self.mse_loss(torch.Tensor(sampled), torch.Tensor(targets))
        mse = torch.mean(mse)

        losses['trainloss/Training MSE: '] = mse
        losses['valloss/Validation MSE: '] = mse

        return losses, None, None


class GPT2Flat_masked(GPT2Flat_fullattention):
    def build_model(self, args):
        super().build_model(args)
        self.gpt2 = GPT2Model_masked(args.gpt2_config)

    def forward(self, *args, **kwargs):
        # check that the weights are tied
        # emb = self.embeddings.quant_emb.emb[0].weight
        # assert self.head.head[0].weight[0, 0] == emb[0, 0]
        return super().forward(*args, **kwargs)

    def generate2_(self, dataset):
        '''
        GPT2Flat_masked can generate much faster all channels in parallel.
        '''
        self.out_times = 1
        self.head.out_times = 1
        self.embeddings.out_times = 1
        self.use_cache = True
        channels = self.args.num_channels
        ex_shift = self.args.example_shift
        inp_len = self.args.sample_rate
        gen_shift = self.args.generate_shift
        gen_len = self.args.generate_length

        xtrain = dataset.x_train_t
        assert xtrain.shape[2] / ex_shift == 2

        data = xtrain[0, :channels, :-gen_shift].clone()
        zeros = torch.zeros((channels, gen_len), dtype=torch.int16)
        data = torch.cat((data, zeros.to(data.device)), dim=1)

        # select half of label timeseries from each batch
        cond = xtrain[:, -2, :xtrain.shape[2]//2].reshape(-1)
        cond = cond[:data.shape[1]]

        # save cond to result_dir for later
        np.save(os.path.join(self.args.result_dir, 'generate_cond.npy'),
                cond.cpu().numpy())
        cond.unsqueeze_(0).unsqueeze_(0)
        data.unsqueeze_(0)

        print(data.shape)
        print(cond.shape)

        for chunk in range(0, data.shape[2]-inp_len, gen_shift):
            end_ind = chunk+inp_len-gen_shift
            inputs = data[:, :, chunk:end_ind]
            cond_ex = cond[:, :, chunk:end_ind]

            past_kv = None
            for t in range(gen_shift):
                inputs, cond_ex, past_kv = self.generate_channels(
                    data, cond, inputs, cond_ex, past_kv, chunk, t)

            # print progress in percent
            if chunk % 100 == 0:
                percent = chunk/data.shape[2]*100
                print('Progress: {:.2f}%'.format(percent), flush=True)

        data = data[0, :, :].cpu().numpy()
        name = 'generated.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

        # decode to raw data
        data = dataset.reconstruct(data)
        name = 'generated_decoded.mat'
        savemat(os.path.join(self.args.result_dir, name), {'X': data})

    def generate_channels(
            self, data, cond, inputs, cond_ex, past_kv, chunk, t):
        end_ind = chunk+inp_len-gen_shift
        logits, past_kv = self.forward({'inputs': inputs,
                                        'condition': cond_ex,
                                        'past_key_values': past_kv})
        logits = logits.detach()
        inputs = sample(self.args, logits)

        data[:, :, end_ind+t:end_ind+t+1] = inputs
        cond_ex = cond[:, :, end_ind+t:end_ind+t+1]

        self.set_tid(end_ind+t-chunk)

        if t < gen_shift - 1:
            # add separator token
            sep_ids = torch.zeros(1, 1, dtype=torch.int32)
            sep_ids = sep_ids.to(data.device)
            sep_emb = self.embeddings.sep_emb(sep_ids)

            out = self.gpt2(inputs_embeds=sep_emb,
                            past_key_values=past_kv,
                            use_cache=True)
            past_kv = out.past_key_values

            self.set_chid_head(0)
            out = out[0].detach()
            out = self.reshape_output(self.head(out))
            inputs = sample(self.args, out)

        self.set_tid(None)


class GPT2Model_masked(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.wte = None

        hiddens = config.num_hidden_layers
        self.h = ModuleList(
            [GPT2Block_masked(config, layer_idx=i) for i in range(hiddens)])

        self.post_init()


class GPT2Block_masked(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = GPT2Attention_masked(config, layer_idx=layer_idx)


class GPT2Attention_masked(GPT2Attention):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        C = config.num_channels

        # update self.bias to include custom attention mask
        I, J = np.meshgrid(np.arange(self.bias.shape[-2]),
                           np.arange(self.bias.shape[-1]),
                           indexing='ij')

        mask = ((J % C) <= (I % C)) & (J > I - C) & (J % C != 0)
        self.bias[:, :, mask] = False
