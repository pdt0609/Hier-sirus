import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings("ignore")


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment
from encoder import EncodingModel
# import wandb

from transformers import BertTokenizer
from losses import TripletLoss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P) ) # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
#     # Optional: BNInception uses label smoothing, apply it for retraining also
#     # "Rethinking the Inception Architecture for Computer Vision", p. 6
#     import sklearn.preprocessing
#     T = T.cpu().numpy()
#     T = sklearn.preprocessing.label_binarize(
#         T, classes = range(0, nb_classes)
#     )
#     T = T * (1 - smoothing_const)
#     T[T == 0] = smoothing_const / (nb_classes - 1)
#     T = torch.FloatTensor(T).cuda()
#     return T

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    import sklearn.preprocessing
    import torch
    
    # Ensure T is a NumPy array
    if isinstance(T, torch.Tensor):
        T = T.cpu().numpy()  # Convert PyTorch tensor to NumPy array
    elif not isinstance(T, (list, tuple, np.ndarray)):
        raise TypeError(f"Unsupported type for T: {type(T)}")
    
    # Perform label binarization
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    
    # Apply label smoothing
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    
    # Convert back to a PyTorch tensor and move to GPU
    T = torch.FloatTensor(T).cuda()
    
    return T

def generate_ETF(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    
    #print(orth_vec.shape,"orth_vec   shape")
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
    
    
    return etf_vec.T

def generate_orth(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    col_norms = torch.norm(orth_vec, dim=0, keepdim=True)  
    orth_vec_normalized = orth_vec / col_norms  
    return orth_vec.T
def generate_new_ort_mat(old_orth_vec,feat_in, num_classes):
    old_orth_vec=old_orth_vec.T
    old_orth_vec, _ = np.linalg.qr(old_orth_vec)
    new_columns = np.random.randn(feat_in, num_classes)

    for i in range(num_classes):
        for j in range(old_orth_vec.shape[1]):
            new_columns[:, i] -= np.dot(old_orth_vec[:, j], new_columns[:, i]) * old_orth_vec[:, j]
        for j in range(i):
            new_columns[:, i] -= np.dot(new_columns[:, j], new_columns[:, i]) * new_columns[:, j]

        norm = np.linalg.norm(new_columns[:, i])
        if norm > 1e-10:  
            new_columns[:, i] /= norm
    new_orth_vec = np.hstack((old_orth_vec, new_columns))
    new_orth_vec = torch.tensor(new_orth_vec).float()
    new_num_classes = old_orth_vec.shape[1] + num_classes
    # assert torch.allclose(torch.matmul(new_orth_vec.T, new_orth_vec), torch.eye(new_num_classes), atol=1.e-7), \
    #     "The max irregular value is : {}".format(
    #         torch.max(torch.abs(torch.matmul(new_orth_vec.T, new_orth_vec) - torch.eye(new_num_classes))))
    col_norms = torch.norm(new_orth_vec, dim=0, keepdim=True)  
    orth_vec_normalized = new_orth_vec / col_norms  
    return new_orth_vec.T
# a=generate_orth(256,4).T
# l2_norms = torch.norm(a.T, dim=1)

# print("L2 Norms of Vectors:", l2_norms)
# b=generate_new_ort_mat(a,256,6)

# l2_norms = torch.norm(b.T, dim=1)

# print("L2 Norms of Vectors:", l2_norms)
# print(a)
# print(b)

def generate_GOF(orth_vec, level):
    if level==2:
        target_norms = torch.tensor([relation_dict_with_ids[j] for j in range(41)]).float()
        GOF = orth_vec.T * target_norms
    return GOF.T

last_layer = generate_orth(768, 41)


class ProxyNCA(torch.nn.Module):
    def __init__(self, 
                 nb_classes,
                 sz_embedding,
                 smoothing_const=0.1,
                 scaling_x=1,
                 scaling_p=3,
                 level=None,
                 final_taxonomy=None,
                 last_layer=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        torch.nn.Module.__init__(self)
        self.last_layer = last_layer.clone().detach().float().to(device)
        self.level = level
        self.final_taxonomy = final_taxonomy
        if self.level == 2:
            if final_taxonomy is None:
                raise ValueError("final_taxonomy dictionary is required for level=2")
            num_parent_classes = len(final_taxonomy)
            layer = self.next_layer_matrix(last_layer, final_taxonomy)
            layer = generate_new_ort_mat(layer,768,1)
            layer = torch.tensor(layer).float().to(device)
            self.proxies = torch.nn.Parameter(layer)
            self.proxies.requires_grad = False
        else:
            self.proxies = torch.nn.Parameter(last_layer)
            self.proxies.requires_grad = False
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p
    @staticmethod
    def next_layer_matrix(old_matrix, taxonomy):
        """Compute next layer matrix from taxonomy"""
        old_matrix = np.array(old_matrix)
        new_matrix = np.empty((0, old_matrix.shape[1]))
        
        # Sort taxonomy keys to ensure consistent ordering
        sorted_keys = sorted(taxonomy.keys())
        for key in sorted_keys:
            child_indices = taxonomy[key]
            child_vectors = old_matrix[child_indices]
            mean_vector = np.mean(child_vectors, axis=0)
            new_matrix = np.vstack([new_matrix, mean_vector])
            
        return new_matrix
    def forward(self, X, T):
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator

        if X.shape[0] != T.shape[0]:
        # Option 1: Reshape X if it's a multiple of T's batch size
            if X.shape[0] % T.shape[0] == 0:
                factor = X.shape[0] // T.shape[0]
                X = X.view(T.shape[0], factor, -1).mean(1)  # Average across factor
                D = torch.cdist(X, P) ** 2
        
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        
        return loss.mean()




class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        ##
        self.encoder = EncodingModel(self.config)
        #fixing
        self.class_num=41


    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist
    def _cosine_similarity(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        cos = nn.CosineSimilarity(dim=1)
        sim = []
        for i in range(b):
            sim_i = cos(x2, x1[i])
            sim.append(torch.unsqueeze(sim_i, 0))
        sim = torch.cat(sim, 0)
        return sim
    

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)
        #thanh
        # print("***************** Currently use get_memory_proto")
        # print("proto", proto.shape)
        # print("features",features.shape)
        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)
        #tuning
        #print("***************** Currently use select_memory")
        #print("mem_set", len(mem_set))#list
        #print("mem_feas",mem_feas.shape)

        
        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        
    
    def get_cluster_and_centroids(self, embeddings):

        clustering_model = AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average", distance_threshold=0.3).fit(embeddings)
        clusters = clustering_model.fit_predict(embeddings)
        centroids = {}
        for cluster_id in np.unique(clusters):
            if cluster_id not in centroids:
                cluster_embeddings = embeddings[clusters == cluster_id]
                centroid = torch.mean(cluster_embeddings, dim=0)
                centroids[cluster_id] = centroid


        #tuning
        #print("***************** Currently use get_cluster_and_centroids")
        #print("clusters", clusters.shape, clusters)
        #print("centroids:", centroids[0].shape)

        return clusters, centroids

    def train_model(self, encoder, training_data, seen_des, seen_relations, list_seen_des, is_memory=False, step = None, gof=None, seen_relid = None):
        
        torch.cuda.empty_cache()
        proxy = ProxyNCA(self.class_num - (7-step) *5, 768, last_layer = gof).cuda()
        # proxy_nca_layer2 = ProxyNCA(nb_classes=self.class_num - (7-step) *5,sz_embedding=768, level=2, final_taxonomy=cluster_2_id, last_layer = gof, device=device).to(device)


        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)

        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)

        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch

        
        triplet = TripletLoss()
        optimizer.zero_grad()

        relation_2_cluster = {}
        rep_seen_des = []
        relationid2_clustercentroids = {}
        dict = {i: rel_id for i, rel_id in enumerate(seen_relid)}
        relid_to_index = {rel_id: i for i, rel_id in dict.items()}

        for i in range(epoch):         
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                torch.cuda.empty_cache()
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)

                batch_instance = {'ids': [], 'mask': []} 

                batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)

                #tuning
                #print("batch_instance",batch_instance)

                hidden = encoder(instance) # b, dim
                #thanh
                hidden_lv11= encoder(instance, layer=11)
                # print("_______hidden_lv11",hidden_lv11.shape)
                
                
                rep_des = encoder(batch_instance, is_des = True) # b, dim
                
                #tuning

                #print("***********hidden.shape",hidden.shape)
                #print("***********rep_des.shape",rep_des.shape)

                with torch.no_grad():
                    rep_seen_des = []
                    for i2 in range(len(list_seen_des)):
                        sample = {
                            'ids' : torch.tensor([list_seen_des[i2]['ids']]).to(self.config.device),
                            'mask' : torch.tensor([list_seen_des[i2]['mask']]).to(self.config.device)
                        }
                        hidden_des = encoder(sample, is_des=True)
                        hidden_des = hidden_des.detach().cpu().data
                        rep_seen_des.append(hidden_des)
                    rep_seen_des = torch.cat(rep_seen_des, dim=0)
                    clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)
                flag = 0
                if len(clusters) == max(clusters) + 1:
                    flag = 1



                relationid2_clustercentroids = {}
                for index, rel in enumerate(seen_relations):
                    relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]
                #tuning
                # print("relationid2_clustercentroids",relationid2_clustercentroids)


                
                relation_2_cluster = {}

                for i1 in range(len(seen_relations)):
                    relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]


                cluster_2_id = {}

                NUM_RELATIONS = self.class_num - (7-step) *5 -1
                # for relation_id in range(NUM_RELATIONS):
                #     if relation_id not in relation_2_cluster:
                #         relation_2_cluster[relation_id] = NUM_RELATIONS
                        
                for relation_id, cluster_id in relation_2_cluster.items():
                    if cluster_id not in cluster_2_id:
                        cluster_2_id[cluster_id] = []
                    cluster_2_id[cluster_id].append(relation_id)
                
                # Sort the lists for consistency (optional)
                for cluster_id in cluster_2_id:
                    cluster_2_id[cluster_id].sort()


                converted_cluster_2_id = {
                    cluster: [relid_to_index[rel_id] for rel_id in rel_ids]
                    for cluster, rel_ids in cluster_2_id.items()
                }
                # print(converted_cluster_2_id)
                # print("----------relation_2_cluster",relation_2_cluster)
                #thanh
                # print("----------cluster_2_id",cluster_2_id)

                proxy_nca_layer2 = ProxyNCA(nb_classes=self.class_num - (7-step) *5,sz_embedding=768, level=2, final_taxonomy=converted_cluster_2_id, last_layer = gof).cuda()

                max_key = max(relation_2_cluster.keys()) + 1
                mapping = torch.zeros(max_key, dtype=torch.long, device='cuda:0')
                for k, v in relation_2_cluster.items():
                    mapping[k] = v
                new_label = mapping[labels]

                # print("__________new_label",new_label)
                
                loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=0.05,relation_2_cluster=relation_2_cluster)  # Recompute loss2
                loss7 = proxy_nca_layer2(hidden_lv11, new_label)
                    
                cluster_centroids = []

                for label in labels:
                    cluster_centroids.append(relationid2_clustercentroids[label.item()])

                
                
                #tuning
                #print("cluster_centroids",len(cluster_centroids), "size of each", cluster_centroids[0].shape)
                #print("labels", labels)
                
                cluster_centroids  = torch.stack(cluster_centroids, dim = 0).to(self.config.device)
                
                nearest_cluster_centroids = []
                for hid in hidden:
                    cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids, dim=1)

                    try:
                        top2_similarities, top2_indices = torch.topk(cos_similarities, k=2, dim=0)

                        if len(top2_indices) > 1:
                            top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                        else:
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    except RuntimeError as e:
                        print(f"RuntimeError in top-k selection: {e}")
                        top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    nearest_cluster_centroids.append(top2_centroids)
                
                
                #tuning
                #print("nearest_cluster_centroids",len(nearest_cluster_centroids),nearest_cluster_centroids[0].shape)


                
                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)

                #fixing
                target_classes = torch.full(
                torch.Size([16]), 
                fill_value=16, 
                dtype=torch.int64, 
                device=self.config.device)

                #print("hiden",hidden.shape)#16 768
                
                target_classes = labels
                target_classes = torch.tensor([relid_to_index[int(rel_id)] for rel_id in target_classes])

                # print("________target_classes",target_classes)
                

                
                if flag == 0:
                    loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)
                    loss5 = proxy(hidden, target_classes).cuda()
                    loss = loss1 + 2*loss2 + 0.5*loss5 +0.25*loss7
                else:
                    loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)
                    loss = loss1 #+ 2*loss2   
         
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 

                del hidden, rep_des, loss
        print('')             

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total
    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        """
        Args:
            encoder: Encoder
            seen_proto: seen prototypes. NxH tensor
            seen_relid: relation id of protoytpes
            test_data: test data
            rep_des: representation of seen relation description. N x H tensor

        Returns:

        """
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        corrects = 0.0
        corrects1 = 0.0
        corrects2 = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            with torch.no_grad():
                hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            # logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations
            logits = self._cosine_similarity(fea, seen_proto)  # (B, N)
            logits_des = self._cosine_similarity(fea, rep_des)  # (B, N)

            logits_rrf = logits + logits_des 
           
            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size

            # by logits_des
            cur_index1 = torch.argmax(logits_des,dim=1)
            pred1 = []
            for i in range(cur_index1.size()[0]):
                pred1.append(seen_relid[int(cur_index1[i])])
            pred1 = torch.tensor(pred1)
            correct1 = torch.eq(pred1, label).sum().item()
            acc1 = correct1/ batch_size
            corrects1 += correct1

            # by rrf
            cur_index2 = torch.argmax(logits_rrf,dim=1)
            pred2 = []
            for i in range(cur_index2.size()[0]):
                pred2.append(seen_relid[int(cur_index2[i])])
            pred2 = torch.tensor(pred2)
            correct2 = torch.eq(pred2, label).sum().item()
            acc2 = correct2/ batch_size
            corrects2 += correct2

            

            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.write('[EVAL DES] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc1, 100 * (corrects1 / total)) + '\r')
            sys.stdout.write('[EVAL RRF] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc2, 100 * (corrects2 / total)) + '\r')
            sys.stdout.flush()
        print('')
        return corrects / total, corrects1 / total, corrects2 / total

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        
        #tunning
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        
        
        
        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = self.encoder


        #tuning
        #print("-----------------",encoder)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc1, total_acc1 = [], []
        cur_acc2, total_acc2 = [], []


        cur_acc_num, total_acc_num = [], []
        cur_acc_num1, total_acc_num1 = [], []
        cur_acc_num2, total_acc_num2 = [], []


        memory_samples = {}
        data_generation = []
        seen_des = {}
        #thanh
        gof=[]


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])



        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
            torch.cuda.empty_cache()
            for rel in current_relations:
                ids = self.tokenizer.encode(seen_descriptions[rel][0],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.config.max_length)        
                # mask
                mask = np.zeros(self.config.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                mask[:end_index + 1] = 1 
                if rel not in seen_des:
                    seen_des[rel] = {}
                    seen_des[rel]['ids'] = ids
                    seen_des[rel]['mask'] = mask

            # get representation of seen description
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])


            #thanh
            # print("___seen_relid",seen_relid)

            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]

            list_seen_des = []
            for i in range(len(seen_relations)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []

            if step > 0:
                relations = list(set(seen_relations) - set(current_relations))
                for rel in relations:
                    training_data_initialize += memory_samples[rel]            

            for rel in current_relations:
                training_data_initialize += training_data[rel]



            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)
            print("---------seen_proto",seen_proto.shape)


            
            #thanh
            if step == 0:
                print("GOF initialization in TASK 0")
                gof=generate_orth(768, 6)
            else:
                print("GOF initialization in TASK %d" % step)
                #thanh
                old_gof=gof
                new_gof=generate_new_ort_mat(gof,768,100)
                # gof=generate_new_ort_mat(gof,768,5)

                dists = torch.cdist(new_gof, seen_proto, p=2)  
                min_dists, _ = torch.min(dists, dim=1)  
                top5_indices = torch.topk(min_dists, 5).indices  # (5,)
                top5_vectors = new_gof[top5_indices]  # (5, 768)
                gof = torch.cat([old_gof, top5_vectors], dim=0) 



            
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=False, step = step, gof=gof, seen_relid=seen_relid)



            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            
            with torch.no_grad():
                encoder.eval()
                rep_des = []
                for i in range(len(list_seen_des)):
                    sample = {
                        'ids' : torch.tensor([list_seen_des[i]['ids']]).to(self.config.device),
                        'mask' : torch.tensor([list_seen_des[i]['mask']]).to(self.config.device)
                    }
                    hidden = encoder(sample, is_des=True)
                    hidden = hidden.detach().cpu().data
                    rep_des.append(hidden)
                rep_des = torch.cat(rep_des, dim=0)
            encoder.train()

            ac1,ac1_des, ac1_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur,rep_des)
            ac2,ac2_des, ac2_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen, rep_des)
            
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

            cur_acc_num1.append(ac1_des)
            total_acc_num1.append(ac2_des)
            cur_acc1.append('{:.4f}'.format(ac1_des))
            total_acc1.append('{:.4f}'.format(ac2_des))
            print('cur_acc des: ', cur_acc1)
            print('his_acc des: ', total_acc1)

            cur_acc_num2.append(ac1_rrf)
            total_acc_num2.append(ac2_rrf)
            cur_acc2.append('{:.4f}'.format(ac1_rrf))
            total_acc2.append('{:.4f}'.format(ac2_rrf))
            print('cur_acc rrf: ', cur_acc2)
            print('his_acc rrf: ', total_acc2)


        torch.cuda.empty_cache()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        torch.cuda.empty_cache()

        return total_acc_num, total_acc_num1, total_acc_num2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="Tacred", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    acc_list1 = []
    aac_list2 = []

    # for i in range(round_want_to_start, config.total_round, 1):
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc, acc1, aac2 = manager.train()
        acc_list.append(acc)
        acc_list1.append(acc1)
        aac_list2.append(aac2)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))
    accs1 = np.array(acc_list1)
    ave1 = np.mean(accs1, axis=0)
    print('his_acc des mean: ', np.around(ave1, 4))
    accs2 = np.array(aac_list2)
    ave2 = np.mean(accs2, axis=0)
    print('his_acc rrf mean: ', np.around(ave2, 4))