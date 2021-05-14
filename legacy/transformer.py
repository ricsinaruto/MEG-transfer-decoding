import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss, Module, Linear, CrossEntropyLoss, Softmax
from sklearn.decomposition import PCA
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, BertConfig, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from cichy.utils import load_data


class BertModelTS(Module):
    def __init__(self, vocab, seq_len):
        super(BertModelTS, self).__init__()
        self.config = BertConfig(vocab_size=1,
                                 hidden_size=273,
                                 num_hidden_layers=4,
                                 num_attention_heads=3,
                                 intermediate_size=273*4,
                                 max_position_embeddings=seq_len,
                                 position_embedding_type='relative')
        self.model = BertModel(self.config)

        self.weights = torch.tensor(np.random.randn(seq_len), requires_grad=True).float().cuda()
        self.weights = self.weights.view(-1, 1)

        self.output_layer = Linear(self.config.hidden_size, self.config.hidden_size, bias=True)

    def forward(self, x):
        outputs = self.model(inputs_embeds=x[:, :-1, :])[0]
        # missing activation function
        outputs = self.output_layer(outputs[:, 0, :])

        return torch.squeeze(outputs)

    def _forward(self, x):
        outputs = self.model(inputs_embeds=x)[0]
        outputs = outputs.permute(0, 2, 1)
        outputs = torch.matmul(outputs, self.weights)

        return torch.squeeze(outputs)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        pickle.dump(self.weights.cpu(), open(path + 'output_weights', 'wb'))

class BertModelCLS(Module):
    def __init__(self, vocab, seq_len, embed_size):
        super(BertModelCLS, self).__init__()
        self.config = BertConfig(vocab_size=1,
                                 hidden_size=embed_size,
                                 num_hidden_layers=2,
                                 num_attention_heads=3,
                                 intermediate_size=embed_size*2,
                                 max_position_embeddings=seq_len+1,
                                 hidden_dropout_prob=0.5,
                                 attention_probs_dropout_prob=0.5,
                                 position_embedding_type='absolute')
        self.model = BertModel(self.config)

        self.cls_embedding = torch.tensor(np.random.randn(self.config.hidden_size), requires_grad=True).float().cuda()
        self.cls_embedding = self.cls_embedding.view(1, -1)

        self.output_layer = Linear(self.config.hidden_size, 118, bias=False)
        # apparently softmax is not needed

    def forward(self, x):
        embeddings = torch.stack([self.cls_embedding for i in range(x.shape[0])])
        x = torch.cat((embeddings, x), axis=1)
        x = self.model(inputs_embeds=x)[0]
        x = x[:, 0, :]
        x = self.output_layer(x)
        return x

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        pickle.dump(self.cls_embedding.cpu(), open(path + 'cls_embedding', 'wb'))
        pickle.dump(self.output_layer.weight.cpu(), open(path + 'output_weights', 'wb'))


class GPT2LM(Module):
    def __init__(self, vocab, seq_len, embed_size):
        super(GPT2LM, self).__init__()
        self.config = GPT2Config(vocab_size=vocab,
                                 n_positions=seq_len,
                                 n_ctx=seq_len,
                                 n_embd=embed_size,
                                 n_layer=4,
                                 n_head=4)

        self.model = GPT2LMHeadModel(self.config)

    def forward(self, batch):
        x = self.model(batch[:, :-1])[0]
        return x[:, -1, :], batch[:, -1]

    def save_pretrained(self, path):
        self.model.save_pretrained(path)


class GPT2Raw(Module):
    def __init__(self):
        super(BertModelTS, self).__init__()
        self.config = GPT2Config(vocab_size=self.vocab,
                                 n_positions=self.seq_len,
                                 n_ctx=self.seq_len,
                                 n_embd=273,
                                 n_layer=4,
                                 n_head=3)

    def forward(self, x):
        x = torch.maximum(self.constant, x)
        return x


class Experiment():
    def __init__(self):
        self.random_state = 69
        self.dataset_path = os.path.join('cichy', 'data', 'subj01', 'full_preprocessed')
        self.model_path = os.path.join('cichy', 'results', 'transformer', 'bert_cls')
        self.seq_len = 256
        self.vocab = 1
        self.batch_size = 64
        self.num_trials = 30
        self.num_classes = 118
        self.n_epochs = 5000
        self.load_kmeans = False
        self.load_model = False
        self.do_pca = True
        self.pca_components = 294
        self.lm = False
        criterion_class = CrossEntropyLoss
        model_class = BertModelCLS
        learning_rate = 0.00005

        self.load_cichy_data()
        if self.lm:
            self.quantize()

        if self.load_model:
            self.model = model_class.from_pretrained(self.model_path).cuda()
        else:
            self.model = model_class(self.vocab, self.seq_len, self.pca_components).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion_class().cuda()

        print('Num parameters: ', sum([param.numel() for param in self.model.parameters()]))

    def load_data(self):
        raw = pickle.load(open(self.dataset_path,'rb'))
        raw = raw.transpose()
        raw = (raw - np.mean(raw, axis=0))/np.std(raw, axis=0)

        if self.do_pca:
            raw = PCA(n_components=self.pca_components, random_state=self.random_state).fit_transform(raw)
            #raw = (raw - np.mean(raw, axis=0))/np.std(raw, axis=0)

        self.raw = torch.Tensor(raw).float()
        timesteps = int(self.raw.shape[0]*0.8) # validation split
        self.raw_train = self.raw[:timesteps, :]
        self.raw_valid = self.raw[timesteps:, :]
        if not self.lm:
            self.data_train = self.raw_train.cuda()
            self.data_valid = self.raw_valid.cuda()

    def load_cichy_data(self):
        x_train, y_train, num_examples, trial_list = load_data(self.dataset_path,
                                                 permute=False,
                                                 conditions=118,
                                                 num_components=0,
                                                 resample=4,
                                                 tmin=0,
                                                 tmax=-1,
                                                 remove_epochs=False)
        self.seq_len = x_train.shape[2]
        x_train = x_train.transpose(1, 0, 2).reshape(self.pca_components, -1)
        x_train = (x_train - np.mean(x_train, axis=1))/np.std(x_train, axis=1)
        x_train = x_train.reshape(self.pca_components, -1, self.seq_len).transpose(1, 2, 0)

        num_classes = self.num_classes
        trials = self.num_trials
        lower = int(trials/5)
        upper = int(2*trials/5)
        self.data_valid = torch.Tensor(np.array(np.concatenate(tuple([x_train[lower+trials*i:upper+trials*i] for i in range(num_classes)])))).cuda()
        self.data_train = torch.Tensor(np.array(np.concatenate(tuple([x_train[:lower]] + [x_train[upper+trials*i:lower+trials*(i+1)] for i in range(num_classes)])))).cuda()
        self.y_valid = torch.tensor(np.array(np.concatenate(tuple([y_train[lower+trials*i:upper+trials*i] for i in range(num_classes)]))).reshape(-1), dtype=torch.long).cuda()
        self.y_train = torch.tensor(np.array(np.concatenate(tuple([y_train[:lower]] + [y_train[upper+trials*i:lower+trials*(i+1)] for i in range(num_classes)]))).reshape(-1), dtype=torch.long).cuda()


    def quantize(self):
        if self.load_kmeans:
            self.kmeans = pickle.load(open(os.path.join(self.model_path, 'quantizer'), 'rb'))
            train_indexes = self.kmeans.predict(self.raw_train)
        else:
            self.kmeans = MiniBatchKMeans(self.vocab, random_state=self.random_state)
            train_indexes = self.kmeans.fit_predict(self.raw_train)
            pickle.dump(self.kmeans, open(os.path.join(self.model_path, 'quantizer'), 'wb'))

        valid_indexes = self.kmeans.predict(self.raw_valid)
        self.data_train = torch.Tensor(train_indexes).long().cuda()
        self.data_valid = torch.Tensor(valid_indexes).long().cuda()

    def training_cichy(self):
        best_loss = 1000000
        num_batches = int(self.data_train.shape[0]/self.batch_size) - 1
        for epoch in range(self.n_epochs):
            self.eval_cichy()
            #self.generate_raw()

            self.model.train()
            losses = []
            
            for i in tqdm(range(num_batches + 1)):
                batch = self.data_train[i * self.batch_size : (i+1) * self.batch_size]
                # might need position ids
                outputs = self.model(batch)
                loss = self.criterion(outputs, self.y_train[i * self.batch_size : (i+1) * self.batch_size])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
                
            self.model.save_pretrained(self.model_path)

            loss = sum(losses)/len(losses)
            print('Train loss: ', loss)
            if loss < best_loss:
                best_loss = loss

    def eval_cichy(self):
        self.model.eval()
        num_batches = int(self.data_valid.shape[0]/self.batch_size) - 1

        losses = []
        for i in range(num_batches + 1):
            batch = self.data_valid[i * self.batch_size : (i+1) * self.batch_size]
            # might need position ids
            outputs = self.model(batch)
            loss = self.criterion(outputs, self.y_valid[i * self.batch_size : (i+1) * self.batch_size])
            losses.append(loss.item())

        loss = sum(losses)/len(losses)
        print('Valid loss: ', loss)

    def training(self):
        best_loss = 1000000
        num_batches = int((self.data_train.shape[0] - self.seq_len - 1)/self.batch_size) - 1
        for epoch in range(self.n_epochs):
            self.evaluate()
            #self.generate_raw()

            self.model.train()
            losses = []
            
            for i in tqdm(range(num_batches + 1)):
                batch = [self.data_train[i * self.batch_size + b : i * self.batch_size + b + self.seq_len + 1] for b in range(self.batch_size)]
                batch = torch.stack(batch)
                # might need position ids
                outputs, targets = self.model(batch)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
                
            self.model.save_pretrained(self.model_path)

            loss = sum(losses)/len(losses)
            print('Train loss: ', loss)
            if loss < best_loss:
                best_loss = loss

    def evaluate(self):
        self.model.eval()
        num_batches = int((self.data_valid.shape[0] - self.seq_len - 1)/self.batch_size) - 1

        output_list = []
        loss_list = []
        for i in range(num_batches + 1):
            batch = [self.data_valid[i * self.batch_size + b : i * self.batch_size + b + self.seq_len] for b in range(self.batch_size)]
            batch = torch.stack(batch)

            outputs = self.model(batch)[0].detach()
            targets = self.data_valid[self.seq_len + 1:self.seq_len + 1 + len(outputs)]
            loss = self.criterion(outputs, targets)
            loss_list.append(loss.item())

            outputs = outputs.cpu().numpy()
            if self.lm:
                outputs = np.amax(outputs, axis=1)
                outputs = [self.kmeans.cluster_centers_[int(i)] for i in outputs]
                outputs = np.array(outputs)
            output_list.append(outputs)

        outputs = np.concatenate(tuple(output_list), 0)
        print('Valid loss: ', sum(loss_list)/len(loss_list))

        outputs = outputs[:2000, :]
        if self.lm:
            self.plot_lm(outputs, 'timeseries')
        else:
            self.plot(outputs, 'timeseries')

    def generate_lm(self):
        self.model.eval()
        generation = []
        init = self.model(self.data_valid[:self.seq_len].view(1, self.seq_len))
        init = np.amax(init.detach().cpu().numpy(), axis=1)
        generation.append(init)

        inputs = torch.cat((self.data_valid[1:self.seq_len, :], init.view(1, -1)), 0)
        for i in range(self.seq_len*5):
            output = self.model(inputs.view(1, self.seq_len, -1))
            output = output.detach()
            generation.append(output)
            inputs = torch.cat((inputs[1:, :], output.view(1, -1)), 0)

        generation = torch.stack(generation).cpu().numpy()

        self.plot_lm(generation, 'generated_timeseries')

    def generate_raw(self):
        self.model.eval()
        generation = []
        init = self.model(self.data_valid[:self.seq_len].view(1, self.seq_len, -1))
        init = init.detach()
        generation.append(init)

        inputs = torch.cat((self.data_valid[1:self.seq_len, :], init.view(1, -1)), 0)
        for i in range(self.seq_len*5):
            output = self.model(inputs.view(1, self.seq_len, -1))
            output = output.detach()
            generation.append(output)
            inputs = torch.cat((inputs[1:, :], output.view(1, -1)), 0)

        generation = torch.stack(generation).cpu().numpy()

        self.plot(generation, 'generated_timeseries')

    def plot_lm(self, output, filename):
        indexes = self.data_valid[self.seq_len + 1:self.seq_len + 1 + len(output)].cpu().numpy()
        quantized = [self.kmeans.cluster_centers_[i] for i in indexes]
        quantized = np.array(quantized)

        for ch in [2, 20, 30, 40, 70]:
            plt.subplot(311)
            plt.plot(self.raw_valid[self.seq_len + 1:self.seq_len + 1 + len(output), ch], linewidth=1.0)
            plt.subplot(312)
            plt.plot(quantized[:, ch], linewidth=1.0)
            plt.subplot(313)
            plt.plot(output[:, ch], linewidth=1.0)
            path = os.path.join(self.model_path, filename + str(ch) + '.svg')
            plt.savefig(path, format='svg', dpi=1200)
            plt.close('all')
        
    def plot(self, output, filename):
        for ch in [2, 20, 100, 150, 200]:
            #plt.subplot(211)
            plt.plot(self.raw_valid[self.seq_len + 1:self.seq_len + 1 + len(output), ch].cpu().numpy(), linewidth=1.0)
            #plt.subplot(212)
            plt.plot(output[:, ch], linewidth=1.0)
            path = os.path.join(self.model_path, filename + str(ch) + '.svg')
            plt.savefig(path, format='svg', dpi=1200)
            plt.close('all')


    def evaluate_lm(self):
        self.model.eval()
        num_batches = int((self.data_valid.shape[0] - self.seq_len - 1)/self.batch_size) - 1

        output_list = []
        for i in range(num_batches + 1):
            batch = [self.data_valid[i * self.batch_size + b : i * self.batch_size + b + self.seq_len] for b in range(self.batch_size)]
            batch = torch.stack(batch)

            outputs = self.model(batch)
            output_list.append(outputs.detach())

        outputs = torch.cat(tuple(output_list), 0)
        targets = self.data_valid[self.seq_len + 1:self.seq_len + 1 + len(outputs), :]


        outputs = np.amax(outputs.detach().cpu().numpy(), axis=1)
        quantized_output = [self.kmeans.cluster_centers_[int(i)] for i in outputs]

        indexes = self.indexes_valid[1000:1000+self.seq_len].cpu().numpy()
        quantized = [self.kmeans.cluster_centers_[i] for i in indexes]
        quantized = np.array(quantized)

        for ch in [2, 20, 30, 40, 70]:
            plt.subplot(311)
            plt.plot(self.raw_valid[1000:1000+self.seq_len,ch].cpu().numpy())
            plt.subplot(312)
            plt.plot(quantized[:, ch])
            #plt.subplot(313)
            #plt.plot(quantized_output[:][ch])
            plt.savefig('results/transformer/timeseries' + str(ch) + '.svg', format='svg', dpi=1200)
            plt.close('all')


def main():
    exp = Experiment()
    exp.training_cichy()


if __name__ == "__main__":
    main()