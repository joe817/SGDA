# coding=utf-8
import os

from argparse import ArgumentParser
from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
import math
from sklearn.metrics import f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acmv9')
parser.add_argument("--target", type=str, default='citationv1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3) #5e-3
parser.add_argument("--weight_decay", type=float, default=1e-3) #2e-3
parser.add_argument("--drop_out", type=float, default=1e-1)

parser.add_argument("--perturb", type=bool, default=True)
parser.add_argument("--perturb_value", type=float, default=0.5)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--label_rate", type=float, default=0.05)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim
use_perturb = args.perturb
perturb_value = args.perturb_value
label_rate = args.label_rate

id = "source: {}, target: {}, seed: {}, label_rate:{:.2f}, lr: {}, wd:{}, perturb:{:.3f}, dim: {}" \
    .format(args.source, args.target, seed, label_rate, args.learning_rate, args.weight_decay, perturb_value,
            encoder_dim)
print(id)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = False

dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
source_data.num_classes = dataset.num_classes
print(source_data)

dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
target_data.num_classes = dataset.num_classes
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)


source_train_size = int(source_data.size(0) * label_rate)
label_mask = np.array([1] * source_train_size + [0] * (source_data.size(0) - source_train_size)).astype(bool)
np.random.shuffle(label_mask)
label_mask = torch.tensor(label_mask).to(device)


def index2dense(edge_index,nnode=2708):
    indx = edge_index.cpu().detach().numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj

class add_perturb(nn.Module):
    def __init__(self, dim1, dim2, beta):
        super(add_perturb, self).__init__()
        self.perturb = nn.Parameter(torch.FloatTensor(dim1, dim2).normal_(-beta, beta).to(device))
        self.perturb.requires_grad_(True)

    def forward(self, input):
        return input + self.perturb


class GNN(torch.nn.Module):
    def __init__(self, base_model=None, **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(args.drop_out) for _ in weights]

        self.perturb_layers = nn.ModuleList([
            add_perturb(source_data.size(0), encoder_dim, perturb_value),
            add_perturb(source_data.size(0), encoder_dim, perturb_value)
        ])

        self.conv_layers = nn.ModuleList([
            PPMIConv(dataset.num_features, encoder_dim,
                      weight=weights[0],
                      bias=biases[0],
                      **kwargs),
            PPMIConv(encoder_dim, encoder_dim,
                      weight=weights[1],
                      bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name, perturb):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if perturb:
                x = self.perturb_layers[i](x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

def encode(data, cache_name, perturb=False, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name, perturb)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output

def predict(data, cache_name, perturb=False, mask=None):
    encoded_output = encode(data, cache_name, perturb, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return accuracy, macro_f1, micro_f1


def test(data, cache_name, perturb=False, mask=None):
    for model in models:
        model.eval()
    encoded_output = encode(data, cache_name, perturb)
    logits = predict(data, cache_name, perturb, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy, macro_f1, micro_f1 = evaluate(preds, labels)
    return accuracy, macro_f1, micro_f1, encoded_output

def get_renode_weight(data, pseudo_label):

    ppr_matrix = data.new_adj  
    gpr_matrix = []
    for iter_c in range(data.num_classes):
        iter_gpr = torch.mean(ppr_matrix[pseudo_label==iter_c],dim=0).squeeze()
        gpr_matrix.append(iter_gpr)
    gpr_matrix = torch.stack(gpr_matrix,dim=0).transpose(0,1)

    base_w  = 0.8
    scale_w = 0.4
    nnode = ppr_matrix.size(0)

    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix =  torch.mm(ppr_matrix,gpr_matrix) - torch.mm(ppr_matrix,gpr_rn)/(data.num_classes-1.0)

    label_matrix = F.one_hot(pseudo_label, gpr_matrix.size(1)).float() 
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    
    #computing the ReNode Weight
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=True)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]
    
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(nnode-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
   
    return rn_weight


loss_func = nn.CrossEntropyLoss().to(device)

encoder = GNN().to(device)

cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)


domain_model = nn.Sequential(
    GRL(),
    nn.Linear(encoder_dim, 64),
    nn.ReLU(),
    nn.Dropout(args.drop_out),
    nn.Linear(64, 2),
).to(device)


encoded_source = encode(source_data, args.source)
encoded_target = encode(target_data, args.target)

with open ('tmp/'+args.source+'.pkl', 'rb') as f:
    source_edge_index, norm = pickle.load(f)
with open ('tmp/'+args.target+'.pkl', 'rb') as f:
    target_edge_index, norm = pickle.load(f)

source_data.new_adj = index2dense(source_edge_index, source_data.num_nodes).to(device)
target_data.new_adj = index2dense(target_edge_index, target_data.num_nodes).to(device)


models = [encoder, cls_model, domain_model]
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

epochs = 200


def Entropy(input, weight, label):
    softmax_out = nn.Softmax(dim=-1)(input)
    entropy = -label * torch.log(softmax_out + 1e-5)
    entropy_loss = torch.mean(weight * torch.sum(entropy, dim=1))

    msoftmax = softmax_out.mean(dim=0)
    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    
    return entropy_loss 

def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    global rate
    rate = min((epoch + 1) / epochs, 0.05)

    encoded_source = encode(source_data, args.source, use_perturb)
    encoded_target = encode(target_data, args.target)
    source_logits = cls_model(encoded_source)
    target_logits = cls_model(encoded_target)

    # classifier loss:
    cls_loss = loss_func(source_logits[label_mask], source_data.y[label_mask]) 

    # pseudo labeling loss:
    _, s_plabel = torch.max(source_logits, dim=1)
    s_plabel[label_mask] = source_data.y[label_mask]
    _, t_plabel = torch.max(target_logits, dim=1)

    s_weight = get_renode_weight(source_data, s_plabel).to(device)
    t_weight = get_renode_weight(target_data, t_plabel).to(device)

    s_plabel = F.one_hot(s_plabel, source_data.num_classes)
    t_plabel = F.one_hot(t_plabel, target_data.num_classes)
    semi_loss = Entropy(source_logits[~label_mask], s_weight[~label_mask], s_plabel[~label_mask]) + \
                            Entropy(target_logits, t_weight, t_plabel)
    

    # DA loss
    source_domain_preds = domain_model(encoded_source)
    target_domain_preds = domain_model(encoded_target)

    source_domain_cls_loss = loss_func(
        source_domain_preds,
        torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
    )
    target_domain_cls_loss = loss_func(
        target_domain_preds,
        torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
    )
    loss_grl = source_domain_cls_loss + target_domain_cls_loss



    loss = cls_loss  + loss_grl + (float(epoch) / epochs) * semi_loss#


    optimizer.zero_grad()
    loss.backward()

    if use_perturb:
        for pi in encoder.perturb_layers:
            x_perturb_data = pi.perturb.detach() - args.learning_rate * pi.perturb.grad.data.detach()/torch.norm(pi.perturb.grad.detach(), p ='fro')
            pi.perturb.data = x_perturb_data.data
            pi.perturb.grad[:] = 0

    optimizer.step()


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_macro_f1 = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct, _, _, output_source = test(source_data, args.source, use_perturb, source_data.test_mask)
    target_correct, macro_f1, micro_f1, output_target = test(target_data, args.target)
    print("Epoch: {}, source_acc: {}, target_acc: {}, macro_f1: {}, micro_f1: {}".format(epoch, source_correct,
                                                                                         target_correct, macro_f1,
                                                                                         micro_f1))
    if source_correct > best_source_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_macro_f1 = macro_f1
        best_micro_f1 = micro_f1
        best_epoch = epoch
        with open ('log/{}_{}_embeddings.pkl'.format(args.source, args.target),'wb') as f:
            pickle.dump([output_source.cpu().detach().numpy(), output_target.cpu().detach().numpy()], f)
print("=============================================================")
line = "{}\n - Epoch: {}, best_source_acc: {}, best_target_acc: {}, best_macro_f1: {}, best_micro_f1: {}" \
    .format(id, best_epoch, best_source_acc, best_target_acc, best_macro_f1, best_micro_f1)

print(line)


with open("log/{}-{}.log".format(args.source, args.target), 'a') as f:
    line = "{} - Epoch: {:0>3d}, best_macro_f1: {:.5f}, best_micro_f1: {:.5f}\t" \
               .format(id, best_epoch, best_macro_f1, best_micro_f1) + time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n"
    f.write(line)
