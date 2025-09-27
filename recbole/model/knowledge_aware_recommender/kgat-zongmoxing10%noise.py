# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
KGAT
##################################################
Reference:
    Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in SIGKDD 2019.

Reference code:
    https://github.com/xiangwang1223/knowledge_graph_attention_network
"""
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import torch.nn.functional as F
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, logmap0
from abc import ABC, abstractmethod
from typing import Tuple
from utils.euclidean import givens_rotations, givens_reflection

from torch.utils.data import WeightedRandomSampler


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Aggregator(nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float64)
            self.W2 = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float64)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):

        norm_matrix = (
            norm_matrix.to(torch.float32)
            if norm_matrix.dtype == torch.float64
            else norm_matrix
        )
        ego_embeddings = (
            ego_embeddings.to(torch.float32)
            if ego_embeddings.dtype == torch.float64
            else ego_embeddings
        )

        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":

            # 在模型定义或forward函数内部，确保ego_embeddings是double类型
            ego_embeddings = ego_embeddings.to(torch.double)

            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class KGAT(KnowledgeRecommender):
    r"""KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
    items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGAT, self).__init__(config, dataset)

        # load dataset info
        self.ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
        self.all_hs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").row
        ).to(self.device)
        self.all_ts = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").col
        ).to(self.device)
        self.all_rs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").data
        ).to(self.device)
        self.matrix_size = torch.Size(
            [self.n_users + self.n_entities, self.n_users + self.n_entities]
        )

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.layers = [self.embedding_size] + config["layers"]
        self.aggregator_type = config["aggregator_type"]
        self.mess_dropout = config["mess_dropout"]
        self.reg_weight = config["reg_weight"]

        # generate intermediate data
        self.A_in = (
            self.init_graph()
        )  # init the attention matrix by the structure of ckg
        self.A_in_1 = self.A_in
        self.A_in_2 = self.A_in
        self.A_in_3 = self.A_in

        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = sum(self.layers)
        print("inner size:", inner_size)
        self.projection_head.append(
            torch.nn.Linear(inner_size, inner_size * 4, bias=False, dtype=torch.float64)
        )
        self.projection_head.append(
            torch.nn.BatchNorm1d(
                inner_size * 4, eps=1e-12, affine=affine, dtype=torch.float64
            )
        )
        self.projection_head.append(
            torch.nn.Linear(inner_size * 4, inner_size, bias=False, dtype=torch.float64)
        )
        self.projection_head.append(
            torch.nn.BatchNorm1d(
                inner_size, eps=1e-12, affine=affine, dtype=torch.float64
            )
        )
        self.mode = 0

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # self.user_embedding1 = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )
        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_entity_e"]

        # 添加参数

        self.data_type = torch.double

        self.sizes = (self.n_entities, self.n_relations, self.n_entities)
        self.rank = self.embedding_size
        # self.dropout = dropout
        # self.bias = bias
        self.init_size = 0.001
        self.gamma = 1.0
        self.gamma = nn.Parameter(torch.Tensor([self.gamma]), requires_grad=False)
        # self.entity_embedding1 = nn.Embedding(self.n_entities, self.rank)
        # self.relation_embedding1 = nn.Embedding(self.n_relations, self.rank)
        self.bh = nn.Embedding(self.n_entities, 1)
        self.bh.weight.data = torch.zeros((self.n_entities, 1), dtype=self.data_type)
        self.bt = nn.Embedding(self.n_entities, 1)
        self.bt.weight.data = torch.zeros((self.n_entities, 1), dtype=self.data_type)
        self.new_bt = nn.Embedding(self.n_entities, 1)
        self.new_bh = nn.Embedding(self.n_entities, 1)
        self.new_bh.weight.data = torch.zeros(
            (self.n_entities, 1), dtype=self.data_type
        )
        self.new_bt.weight.data = torch.zeros(
            (self.n_entities, 1), dtype=self.data_type
        )

        self.entity_embedding.weight.data = self.init_size * torch.randn(
            (self.sizes[0], self.rank), dtype=self.data_type
        )
        self.relation_embedding.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        # self.multi_c = args.multi_c
        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)

        self.c = nn.Parameter(c_init, requires_grad=True)

        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.sizes[1] * self.rank)
        self.rel_diag.weight.data = (
            2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        )
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )
        self.act = nn.Softmax(dim=1)

        self.regularizer = N3(weight=1.0)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

        self.scale = torch.Tensor([1.0 / np.sqrt(self.rank)]).double().to(self.device)
        self.bias = "learn"

        # 添加参数
        # self.data_type = torch.float
        self.rank = self.embedding_size
        self.noise_u = torch.randn(
            (self.n_users, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_p = torch.randn(
            (self.n_entities, self.rank * 2), dtype=self.data_type
        ).to(self.device)

        self.noise_u_cts1 = torch.randn(
            (self.n_users, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_u_cts2 = torch.randn(
            (self.n_users, self.rank * 2), dtype=self.data_type
        ).to(self.device)

        self.noise_e_cts1 = torch.randn(
            (self.n_entities, self.rank * 2), dtype=self.data_type
        ).to(self.device)
        self.noise_e_cts2 = torch.randn(
            (self.n_entities, self.rank * 2), dtype=self.data_type
        ).to(self.device)

        self.noise_norm = 0.01

        self.global_counter = 0

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_three_disjoint_indices(self, total_size, subset_size, base_seed=42):

        # 每次调用此函数时递增计数器
        current_seed = base_seed + self.global_counter
        self.global_counter += 1

        # 设置随机种子以确保可复现性
        self.set_seed(current_seed)

        all_indices = torch.arange(total_size)
        shuffled_indices = all_indices[torch.randperm(total_size)]

        ui_rand_samples_1 = shuffled_indices[:subset_size]
        ui_rand_samples_2 = shuffled_indices[subset_size : 2 * subset_size]
        ui_rand_samples_3 = shuffled_indices[2 * subset_size : 3 * subset_size]

        return ui_rand_samples_1, ui_rand_samples_2, ui_rand_samples_3

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl

        adj_list = []

        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False)
                .adjacency_matrix(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            rowsum = np.array(sub_graph.sum(1))

            # 确保 rowsum 不包含0
            epsilon = 1e-8  # 非常小的正数
            rowsum[rowsum == 0] = epsilon

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()

        indices_np = np.array([final_adj_matrix.row, final_adj_matrix.col])
        indices = torch.tensor(indices_np, dtype=torch.long)

        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    # def forward(self):

    def forward_1(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def forward_2(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def forward_3(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_3, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def forward_1_noise(self):
        ego_embeddings = self._get_ego_embeddings()
        x = 0.01 + torch.zeros(
            ego_embeddings.shape[0], ego_embeddings.shape[1], dtype=self.data_type
        ).to(ego_embeddings.device)
        noise = torch.normal(
            mean=torch.tensor([0.0]).to(ego_embeddings.device), std=x
        ).to(ego_embeddings.device)
        ego_embeddings = ego_embeddings + noise
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def forward_2_noise(self):
        ego_embeddings = self._get_ego_embeddings()
        x = 0.01 + torch.zeros(
            ego_embeddings.shape[0], ego_embeddings.shape[1], dtype=self.data_type
        ).to(ego_embeddings.device)
        noise = torch.normal(
            mean=torch.tensor([0.0]).to(ego_embeddings.device), std=x
        ).to(ego_embeddings.device)
        ego_embeddings = ego_embeddings + noise
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def forward_3_noise(self):
        ego_embeddings = self._get_ego_embeddings()
        x = 0.01 + torch.zeros(
            ego_embeddings.shape[0], ego_embeddings.shape[1], dtype=self.data_type
        ).to(ego_embeddings.device)
        noise = torch.normal(
            mean=torch.tensor([0.0]).to(ego_embeddings.device), std=x
        ).to(ego_embeddings.device)
        ego_embeddings = ego_embeddings + noise
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_3, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def cts_loss(self, z_i, z_j, temp, batch_size):  # B * D    B * D

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # 2B * D

        sim = torch.mm(z, z.T) / temp  # 2B * 2B

        sim_i_j = torch.diag(sim, batch_size)  # B*1
        sim_j_i = torch.diag(sim, -batch_size)  # B*1

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(batch_size)

        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        loss = self.ce_loss(logits, labels)
        return loss

    def projection_head_map(self, state, mode):
        for i, l in enumerate(
            self.projection_head
        ):  # 0: Linear 1: BN (relu)  2: Linear 3:BN (relu)
            if i % 2 != 0:
                if mode == 0:
                    l.train()  # set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()  # set BN to eval mode: use a accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state

    # 假设 self.rand_sample 是一个函数，它接受总样本数、采样数量和是否放回参数，并返回采样索引。
    def improved_sampling(self, embeddings, proportion=0.05):
        """
        根据嵌入矩阵大小按比例采样，并引入随机性和难样本挖掘。

        :param embeddings: 输入的嵌入矩阵
        :param proportion: 采样的比例，默认为5%
        :return: 采样索引
        """
        num_samples = int(embeddings.shape[0] * proportion)

        # 如果需要，可以根据某种标准（如样本难度）来调整采样权重
        weights = torch.ones(embeddings.shape[0])  # 默认均匀权重
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=False
        )
        rand_samples = list(sampler)

        return rand_samples

    def calculate_o_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 添加噪声用户-物品交互
        batch_size = len(user)
        num_noise_interactions = int(batch_size * 0.1)  # 添加10%的噪声交互
        user_ids = list(range(self.n_users))
        item_ids = list(range(self.n_items))

        # 确保不会重复选择已经存在的交互
        existing_interactions = set(zip(user, pos_item))

        noise_user, noise_pos_item, noise_neg_item = [], [], []
        for _ in range(num_noise_interactions):
            while True:
                nu = random.choice(user_ids)
                npi = random.choice(item_ids)
                if (nu, npi) not in existing_interactions:
                    break

            noise_user.append(nu)
            noise_pos_item.append(npi)
            noise_neg_item.append(npi)  # 假设噪声交互没有特定的负样本

        # 将噪声交互添加到原始交互中
        user = torch.cat([user, torch.tensor(noise_user, device=user.device)])
        pos_item = torch.cat(
            [pos_item, torch.tensor(noise_pos_item, device=pos_item.device)]
        )
        neg_item = torch.cat(
            [neg_item, torch.tensor(noise_neg_item, device=neg_item.device)]
        )

        user_all_embeddings, entity_all_embeddings = self.forward_2()

        # 计算推荐损失
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        # u-p对比学习--引入高斯噪声
        user_all_embeddings_n, entity_all_embeddings_n = self.forward_2_noise()

        u_embeddings_n = user_all_embeddings_n[user]
        pos_embeddings_n = entity_all_embeddings_n[pos_item]

        u_embeddings_n = self.projection_head_map(u_embeddings_n, self.mode)
        pos_embeddings_n = self.projection_head_map(pos_embeddings_n, 1 - self.mode)

        ui_cts_loss_n = self.cts_loss(
            u_embeddings_n,
            pos_embeddings_n,
            temp=1.0,
            batch_size=u_embeddings_n.shape[0],
        )

        # 引入对抗噪声
        # u-p对抗学习
        # model.zero_grad()
        u_embeddings_n.retain_grad()
        pos_embeddings_n.retain_grad()

        ui_cts_loss_n.backward(retain_graph=True)
        u_unnormalized_noise_a = u_embeddings_n.grad.clone().detach()
        p_unnormalized_noise_b = pos_embeddings_n.grad.clone().detach()

        # 噪声归一化
        u_norm_a = u_unnormalized_noise_a.norm(p=2, dim=-1)
        u_normalized_noise_a = u_unnormalized_noise_a / (
            u_norm_a.unsqueeze(dim=-1) + 1e-10
        )  # add 1e-10 to avoid NaN

        p_norm_b = p_unnormalized_noise_b.norm(p=2, dim=-1)
        p_normalized_noise_b = p_unnormalized_noise_b / (
            p_norm_b.unsqueeze(dim=-1) + 1e-10
        )  # add 1e-10 to avoid NaN

        self.noise_u = self.noise_norm * u_normalized_noise_a
        self.noise_p = self.noise_norm * p_normalized_noise_b

        # u-p对抗学习
        user_all_embeddings_u, entity_all_embeddings_p = (
            user_all_embeddings_n,
            entity_all_embeddings_n,
        )
        gan_u_cts_embedding = user_all_embeddings_u[user]
        gan_p_cts_embedding = entity_all_embeddings_p[pos_item]
        gan_u_cts_embedding = gan_u_cts_embedding + self.noise_u
        gan_p_cts_embedding = gan_p_cts_embedding + self.noise_p

        gan_u_cts_embedding = self.projection_head_map(gan_u_cts_embedding, self.mode)
        gan_p_cts_embedding = self.projection_head_map(
            gan_p_cts_embedding, 1 - self.mode
        )
        # 计算对抗噪声的对比学习损失
        ui_cts_loss_n1 = self.cts_loss(
            gan_u_cts_embedding,
            gan_p_cts_embedding,
            temp=1.0,
            batch_size=gan_u_cts_embedding.shape[0],
        )

        # projection_head_map函数的模式切换
        self.mode = 1 - self.mode

        loss = mf_loss + self.reg_weight * reg_loss + 0.01 * ui_cts_loss_n1

        return loss

    def calculate_h_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 添加噪声用户-物品交互
        batch_size = len(user)
        num_noise_interactions = int(batch_size * 0.1)  # 添加10%的噪声交互
        user_ids = list(range(self.n_users))
        item_ids = list(range(self.n_items))

        # 确保不会重复选择已经存在的交互
        existing_interactions = set(zip(user, pos_item))

        noise_user, noise_pos_item, noise_neg_item = [], [], []
        for _ in range(num_noise_interactions):
            while True:
                nu = random.choice(user_ids)
                npi = random.choice(item_ids)
                if (nu, npi) not in existing_interactions:
                    break

            noise_user.append(nu)
            noise_pos_item.append(npi)
            noise_neg_item.append(npi)  # 假设噪声交互没有特定的负样本

        # 将噪声交互添加到原始交互中
        user = torch.cat([user, torch.tensor(noise_user, device=user.device)])
        pos_item = torch.cat(
            [pos_item, torch.tensor(noise_pos_item, device=pos_item.device)]
        )
        neg_item = torch.cat(
            [neg_item, torch.tensor(noise_neg_item, device=neg_item.device)]
        )

        user_all_embeddings, entity_all_embeddings = self.forward_1()

        # 计算推荐损失
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        # u-p对比学习--引入高斯噪声
        user_all_embeddings_n, entity_all_embeddings_n = self.forward_1_noise()

        u_embeddings_n = user_all_embeddings_n[user]
        pos_embeddings_n = entity_all_embeddings_n[pos_item]

        u_embeddings_n = self.projection_head_map(u_embeddings_n, self.mode)
        pos_embeddings_n = self.projection_head_map(pos_embeddings_n, 1 - self.mode)

        ui_cts_loss_n = self.cts_loss(
            u_embeddings_n,
            pos_embeddings_n,
            temp=1.0,
            batch_size=u_embeddings_n.shape[0],
        )

        # 引入对抗噪声
        # u-p对抗学习
        # model.zero_grad()
        u_embeddings_n.retain_grad()
        pos_embeddings_n.retain_grad()

        ui_cts_loss_n.backward(retain_graph=True)
        u_unnormalized_noise_a = u_embeddings_n.grad.clone().detach()
        p_unnormalized_noise_b = pos_embeddings_n.grad.clone().detach()

        # 噪声归一化
        u_norm_a = u_unnormalized_noise_a.norm(p=2, dim=-1)
        u_normalized_noise_a = u_unnormalized_noise_a / (
            u_norm_a.unsqueeze(dim=-1) + 1e-10
        )  # add 1e-10 to avoid NaN

        p_norm_b = p_unnormalized_noise_b.norm(p=2, dim=-1)
        p_normalized_noise_b = p_unnormalized_noise_b / (
            p_norm_b.unsqueeze(dim=-1) + 1e-10
        )  # add 1e-10 to avoid NaN

        self.noise_u = self.noise_norm * u_normalized_noise_a
        self.noise_p = self.noise_norm * p_normalized_noise_b

        # u-p对抗学习
        user_all_embeddings_u, entity_all_embeddings_p = (
            user_all_embeddings_n,
            entity_all_embeddings_n,
        )
        gan_u_cts_embedding = user_all_embeddings_u[user]
        gan_p_cts_embedding = entity_all_embeddings_p[pos_item]
        gan_u_cts_embedding = gan_u_cts_embedding + self.noise_u
        gan_p_cts_embedding = gan_p_cts_embedding + self.noise_p

        gan_u_cts_embedding = self.projection_head_map(gan_u_cts_embedding, self.mode)
        gan_p_cts_embedding = self.projection_head_map(
            gan_p_cts_embedding, 1 - self.mode
        )
        # 计算对抗噪声的对比学习损失
        ui_cts_loss_n1 = self.cts_loss(
            gan_u_cts_embedding,
            gan_p_cts_embedding,
            temp=1.0,
            batch_size=gan_u_cts_embedding.shape[0],
        )

        # projection_head_map函数的模式切换
        self.mode = 1 - self.mode

        loss = mf_loss + self.reg_weight * reg_loss + 0.01 * ui_cts_loss_n1

        return loss

    def calculate_e_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 添加噪声用户-物品交互
        batch_size = len(user)
        num_noise_interactions = int(batch_size * 0.1)  # 添加10%的噪声交互
        user_ids = list(range(self.n_users))
        item_ids = list(range(self.n_items))

        # 确保不会重复选择已经存在的交互
        existing_interactions = set(zip(user, pos_item))

        noise_user, noise_pos_item, noise_neg_item = [], [], []
        for _ in range(num_noise_interactions):
            while True:
                nu = random.choice(user_ids)
                npi = random.choice(item_ids)
                if (nu, npi) not in existing_interactions:
                    break

            noise_user.append(nu)
            noise_pos_item.append(npi)
            noise_neg_item.append(npi)  # 假设噪声交互没有特定的负样本

        # 将噪声交互添加到原始交互中
        user = torch.cat([user, torch.tensor(noise_user, device=user.device)])
        pos_item = torch.cat(
            [pos_item, torch.tensor(noise_pos_item, device=pos_item.device)]
        )
        neg_item = torch.cat(
            [neg_item, torch.tensor(noise_neg_item, device=neg_item.device)]
        )

        user_all_embeddings, entity_all_embeddings = self.forward_3()

        # 计算推荐损失
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        # u-p对比学习--引入高斯噪声
        user_all_embeddings_n, entity_all_embeddings_n = self.forward_3_noise()

        u_embeddings_n = user_all_embeddings_n[user]
        pos_embeddings_n = entity_all_embeddings_n[pos_item]

        u_embeddings_n = self.projection_head_map(u_embeddings_n, self.mode)
        pos_embeddings_n = self.projection_head_map(pos_embeddings_n, 1 - self.mode)

        ui_cts_loss_n = self.cts_loss(
            u_embeddings_n,
            pos_embeddings_n,
            temp=1.0,
            batch_size=u_embeddings_n.shape[0],
        )

        # 引入对抗噪声
        # u-p对抗学习
        # model.zero_grad()
        u_embeddings_n.retain_grad()
        pos_embeddings_n.retain_grad()

        ui_cts_loss_n.backward(retain_graph=True)
        u_unnormalized_noise_a = u_embeddings_n.grad.clone().detach()
        p_unnormalized_noise_b = pos_embeddings_n.grad.clone().detach()

        # 噪声归一化
        u_norm_a = u_unnormalized_noise_a.norm(p=2, dim=-1)
        u_normalized_noise_a = u_unnormalized_noise_a / (
            u_norm_a.unsqueeze(dim=-1) + 1e-10
        )  # add 1e-10 to avoid NaN

        p_norm_b = p_unnormalized_noise_b.norm(p=2, dim=-1)
        p_normalized_noise_b = p_unnormalized_noise_b / (
            p_norm_b.unsqueeze(dim=-1) + 1e-10
        )  # add 1e-10 to avoid NaN

        self.noise_u = self.noise_norm * u_normalized_noise_a
        self.noise_p = self.noise_norm * p_normalized_noise_b

        # u-p对抗学习
        user_all_embeddings_u, entity_all_embeddings_p = (
            user_all_embeddings_n,
            entity_all_embeddings_n,
        )
        gan_u_cts_embedding = user_all_embeddings_u[user]
        gan_p_cts_embedding = entity_all_embeddings_p[pos_item]
        gan_u_cts_embedding = gan_u_cts_embedding + self.noise_u
        gan_p_cts_embedding = gan_p_cts_embedding + self.noise_p

        gan_u_cts_embedding = self.projection_head_map(gan_u_cts_embedding, self.mode)
        gan_p_cts_embedding = self.projection_head_map(
            gan_p_cts_embedding, 1 - self.mode
        )
        # 计算对抗噪声的对比学习损失
        ui_cts_loss_n1 = self.cts_loss(
            gan_u_cts_embedding,
            gan_p_cts_embedding,
            temp=1.0,
            batch_size=gan_u_cts_embedding.shape[0],
        )

        # projection_head_map函数的模式切换
        self.mode = 1 - self.mode

        loss = mf_loss + self.reg_weight * reg_loss + 0.01 * ui_cts_loss_n1

        return loss

        # 引入双曲空间
        """
        self.n_entities = sizes[0]
        self.n_relations = sizes[1]
        self.embedding_size = rank
        其中i=0时，h = queries[:, i]
            同理,r = queries[:, 1]
            同理,t =  queries[:, 2]

        pos_t = queries[:, 2]
        c = self.c = nn.Parameter(c_init, requires_grad=True)

        rel_diag = self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        rel_diag.weight.data = self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.init_size = 0.001 = init_size
        torch.float64 = self.data_type
        scale = self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        act = self.act = nn.Softmax(dim=1)
        rel = self.rel = nn.Embedding(sizes[1], rank)
        bh = self.bh = nn.Embedding(sizes[0], 1)


        bt = self.bt = nn.Embedding(self.n_entities, 1)
        bt.weight.data = self.bt.weight.data = torch.zeros((self.n_entities, 1), dtype=self.data_type)

        self.entity_embedding = self.entity = nn.Embedding(self.n_entities, self.embedding_size)
        self.entity_embedding.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)#上面是不是未做这个的定义？


        """

    def get_rhs(self, queries):
        """Get embeddings and biases of target entities."""

        return self.entity_embedding(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return -hyp_distance_multi_c(lhs_e, rhs_e, c) ** 2

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""

        # print("这个数字是：", queries[:, 1])
        # print("这个形状是", queries[:, 1].shape)
        # print("rel_diag(r)", self.rel_diag(queries[:, 1]))
        # print("rel_diag(r).shape", self.rel_diag(queries[:, 1]).shape)

        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity_embedding(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1).to(self.device)
        context_vec = (
            self.context_vec(queries[:, 1]).view((-1, 1, self.rank)).to(self.device)
        )
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        # rel, _ = torch.chunk(self.relation_embedding(queries[:, 1]), 2, dim=1)
        rel = self.relation_embedding(queries[:, 1])
        rel = expmap0(rel, c)

        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

    def score(self, lhs, rhs):

        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e)
        if self.bias == "constant":
            return self.gamma.item() + score
        elif self.bias == "learn":
            return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity_embedding(queries[:, 0])
        rel_e = self.relation_embedding(queries[:, 1])
        rhs_e = self.entity_embedding(queries[:, 2])
        return head_e, rel_e, rhs_e

    def Forward(self, queries):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                             shape is (n_queries x 1) if eval_mode is false
                             else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))

        # get factors for regularization
        factors = self.get_factors(queries)
        return predictions, factors

    def neg_sampling_loss(self, input_batch, neg_samples):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        # positive samples
        positive_score, factors = self.Forward(input_batch)
        positive_score = F.logsigmoid(positive_score)

        # negative samples
        negative_score, _ = self.Forward(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        loss = -torch.cat([positive_score, negative_score], dim=0).mean()
        return loss, factors

    def calculate_loss1(self, input_batch, neg_samples):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """

        loss, factors = self.neg_sampling_loss(input_batch, neg_samples)

        # regularization loss
        loss += self.regularizer.forward(factors)
        return loss

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)

        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = h_e.double()
        r_trans_w = r_trans_w.double()

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _get_rotate_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h)
        pos_t_e = self.entity_embedding(pos_t)
        neg_t_e = self.entity_embedding(neg_t)
        r_e = self.relation_embedding(r)

        # 分割实部与虚部

        h_real = h_e[:, : h_e.shape[1] // 2]  # [batch_size, 1, emb_dim_half]
        h_imag = h_e[:, h_e.shape[1] // 2 :]  # [batch_size, 1, emb_dim_half]
        head_e = h_real, h_imag

        pos_t_real = pos_t_e[:, : pos_t_e.shape[1] // 2]
        pos_t_imag = pos_t_e[:, pos_t_e.shape[1] // 2 :]
        pos_rhs_e = pos_t_real, pos_t_imag

        neg_t_real = neg_t_e[:, : neg_t_e.shape[1] // 2]
        neg_t_imag = neg_t_e[:, neg_t_e.shape[1] // 2 :]
        neg_rhs_e = neg_t_real, neg_t_imag

        rel_e = r_e[:, : r_e.shape[1] // 2], r_e[:, r_e.shape[1] // 2 :]
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = head_e[0] * cos - head_e[1] * sin, head_e[0] * sin + head_e[1] * cos

        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
        # print(lhs_e[0].shape)
        # print(rel_e[0].shape)
        # print(pos_rhs_e[0].shape)
        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')

        return lhs_e, rel_e, pos_rhs_e, neg_rhs_e

    def calculate_kg_h_loss(self, interaction):

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        # 添加噪声三元组
        batch_size = len(h)
        num_noise_triples = int(batch_size * 0.1)  # 添加10%的噪声三元组
        entity_ids = list(range(self.n_entities))
        relation_ids = list(range(self.n_relations))

        # 确保不会重复选择已经存在的三元组
        existing_triples = set(zip(h, r, pos_t))

        noise_h, noise_r, noise_pos_t, noise_neg_t = [], [], [], []
        for _ in range(num_noise_triples):
            while True:
                nh = random.choice(entity_ids)
                nr = random.choice(relation_ids)
                npt = random.choice(entity_ids)
                if (nh, nr, npt) not in existing_triples:
                    break

            noise_h.append(nh)
            noise_r.append(nr)
            noise_pos_t.append(npt)
            noise_neg_t.append(npt)  # 假设噪声三元组没有特定的负样本

        # 将噪声三元组添加到原始三元组中
        h = torch.cat([h, torch.tensor(noise_h, device=h.device)])
        r = torch.cat([r, torch.tensor(noise_r, device=r.device)])
        pos_t = torch.cat([pos_t, torch.tensor(noise_pos_t, device=pos_t.device)])
        neg_t = torch.cat([neg_t, torch.tensor(noise_neg_t, device=neg_t.device)])

        input_batch = torch.stack((h, r, pos_t), dim=1)
        neg_samples = torch.stack((h, r, neg_t), dim=1)

        loss1 = self.calculate_loss1(input_batch, neg_samples)

        return loss1

    def calculate_kg_o_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        # 添加噪声三元组
        batch_size = len(h)
        num_noise_triples = int(batch_size * 0.1)  # 添加10%的噪声三元组
        entity_ids = list(range(self.n_entities))
        relation_ids = list(range(self.n_relations))

        # 确保不会重复选择已经存在的三元组
        existing_triples = set(zip(h, r, pos_t))

        noise_h, noise_r, noise_pos_t, noise_neg_t = [], [], [], []
        for _ in range(num_noise_triples):
            while True:
                nh = random.choice(entity_ids)
                nr = random.choice(relation_ids)
                npt = random.choice(entity_ids)
                if (nh, nr, npt) not in existing_triples:
                    break

            noise_h.append(nh)
            noise_r.append(nr)
            noise_pos_t.append(npt)
            noise_neg_t.append(npt)  # 假设噪声三元组没有特定的负样本

        # 将噪声三元组添加到原始三元组中
        h = torch.cat([h, torch.tensor(noise_h, device=h.device)])
        r = torch.cat([r, torch.tensor(noise_r, device=r.device)])
        pos_t = torch.cat([pos_t, torch.tensor(noise_pos_t, device=pos_t.device)])
        neg_t = torch.cat([neg_t, torch.tensor(noise_neg_t, device=neg_t.device)])

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def calculate_kg_e_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        # 添加噪声三元组
        batch_size = len(h)
        num_noise_triples = int(batch_size * 0.1)  # 添加10%的噪声三元组
        entity_ids = list(range(self.n_entities))
        relation_ids = list(range(self.n_relations))

        # 确保不会重复选择已经存在的三元组
        existing_triples = set(zip(h, r, pos_t))

        noise_h, noise_r, noise_pos_t, noise_neg_t = [], [], [], []
        for _ in range(num_noise_triples):
            while True:
                nh = random.choice(entity_ids)
                nr = random.choice(relation_ids)
                npt = random.choice(entity_ids)
                if (nh, nr, npt) not in existing_triples:
                    break

            noise_h.append(nh)
            noise_r.append(nr)
            noise_pos_t.append(npt)
            noise_neg_t.append(npt)  # 假设噪声三元组没有特定的负样本

        # 将噪声三元组添加到原始三元组中
        h = torch.cat([h, torch.tensor(noise_h, device=h.device)])
        r = torch.cat([r, torch.tensor(noise_r, device=r.device)])
        pos_t = torch.cat([pos_t, torch.tensor(noise_pos_t, device=pos_t.device)])
        neg_t = torch.cat([neg_t, torch.tensor(noise_neg_t, device=neg_t.device)])

        lhs_e, r_e, pos_t_e, neg_t_e = self._get_rotate_embedding(h, r, pos_t, neg_t)

        pos_tail_score = torch.sum(
            lhs_e[0] * pos_t_e[0] + lhs_e[1] * pos_t_e[1], 1, keepdim=True
        )
        neg_tail_score = torch.sum(
            lhs_e[0] * neg_t_e[0] + lhs_e[1] * neg_t_e[1], 1, keepdim=True
        )

        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
        # has_negative = torch.any(pos_tail_score < 0)

        # if has_negative:
        #    print("The pos_tail_score vector contains negative values.")
        # else:
        #    print("The pos_tail_score vector does not contain any negative values.")

        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')

        # pos_tail_score = torch.norm(torch.stack([lhs_e[0] - pos_t_e[0], lhs_e[1] - pos_t_e[1]], dim=0), p=2, dim=0).sum(dim=1,keepdim=True)
        # neg_tail_score = torch.norm(torch.stack([lhs_e[0] - neg_t_e[0], lhs_e[1] - neg_t_e[1]], dim=0), p=2, dim=0).sum(dim=1,keepdim=True)

        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
        # has_negative = torch.any(pos_tail_score < 0)

        # if has_negative:
        #    print("The pos_tail_score vector contains negative values.")
        # else:
        #   print("The pos_tail_score vector does not contain any negative values.")

        # print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')

        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(
            lhs_e[0],
            r_e[0],
            pos_t_e[0],
            neg_t_e[0],
            lhs_e[1],
            r_e[1],
            pos_t_e[1],
            neg_t_e[1],
        )

        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def generate_transE_score3(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding(r)

        head_e = h_e[:, : h_e.shape[1] // 2], h_e[:, h_e.shape[1] // 2 :]
        rhs_e = t_e[:, : t_e.shape[1] // 2], t_e[:, t_e.shape[1] // 2 :]
        rel_e = r_e[:, : r_e.shape[1] // 2], r_e[:, r_e.shape[1] // 2 :]
        # print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')
        # print(head_e[0].shape)
        # print(rel_e[0].shape)
        # print(rhs_e[0].shape)

        # print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')

        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        # lhs_e = [head_e[0] * cos - head_e[1] * sin,head_e[0] * sin + head_e[1] * cos]
        lhs_e = head_e[0] * cos - head_e[1] * sin, head_e[0] * sin + head_e[1] * cos

        # cos_sim1 = nn.CosineSimilarity(dim=1, eps=1e-8)(lhs_e[0], self.tanh(rhs_e[0]))
        # cos_sim2 = nn.CosineSimilarity(dim=1, eps=1e-8)(lhs_e[1], self.tanh(rhs_e[1]))

        # softmax_scores1 = F.softmax(cos_sim1, dim=0)
        # softmax_scores2 = F.softmax(cos_sim2, dim=0)
        # softmax_scores = softmax_scores1 + softmax_scores2

        score = torch.mul(lhs_e[0], self.tanh(rhs_e[0])).sum(dim=1) + torch.mul(
            lhs_e[1], self.tanh(rhs_e[1])
        ).sum(dim=1)
        # score = torch.mul(lhs_e, self.tanh(t_e)).sum(dim=1)
        # score = torch.sum(lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],1, keepdim=True)
        # kg_score = score.squeeze()
        # print('**************************************************')
        # print(kg_score.shape)
        # print('**************************************************')

        return score

    def generate_transE_score1(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings1 = self._get_ego_embeddings()
        h_e1 = all_embeddings1[hs]
        t_e1 = all_embeddings1[ts]
        r_e1 = self.relation_embedding(r)

        # h_e1 = self.entity_embedding1(hs)
        # t_e1 = self.entity_embedding1(ts)
        # r_e1 = self.relation_embedding1(r)

        # r_trans_w = self.trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)

        # h_e = torch.matmul(h_e, r_trans_w)
        # t_e = torch.matmul(t_e, r_trans_w)

        # kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)
        """
        此处参考
        def get_queries(self, queries):用.weight重新进行计算
        ns_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)为什么view维度少了一维？
        目前到了context_vec.weight这一步


        h_e = head
        self.rel.weight[r] = relation_embedding.weight[r]
        .weight[r] = (queries[:, 1])

        """
        # head = self.entity(queries[:, 0])#未修改

        # print("这个是r", r)
        # print("这个是rel_diag.weight[r]", self.rel_diag.weight[r])
        # print("rel_diag.weight[r].shape", self.rel_diag.weight[r].shape)

        c = F.softplus(self.c[r])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(r), 2, dim=1)
        rot_q = givens_rotations(rot_mat, h_e1).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, h_e1).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        # rel, _ = torch.chunk(r_e1, 2, dim=1)

        rel = r_e1

        # rel = r_e
        rel = expmap0(rel, c)

        res = project(mobius_add(lhs, rel, c), c)

        lhs_e = (res, c)
        # lhs_biases = self.bh(hs)
        rhs_e = t_e1
        # rhs_biases = self.bt(ts)

        # score = self.similarity_score(lhs_e, rhs_e) + self.gamma.item()
        # score = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))

        # 第二次训练，换算到欧式空间相似度得分

        # print("*******************这是换算到欧式空间相似度得分的训练****************************")
        res1 = logmap0(res, c)
        rhs_e1 = logmap0(rhs_e, c)

        kg_score = torch.mul(rhs_e1, self.tanh(res1)).sum(dim=1)
        # print('___________________________________________________')
        # print(kg_score.shape)
        # print('___________________________________________________')
        return kg_score

    def generate_transE_score2(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]

        r_trans_w = self.trans_w.weight[r].view(
            self.embedding_size, self.kg_embedding_size
        )

        h_e = h_e.double()
        r_trans_w = r_trans_w.double()

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix"""

        kg_score_list_1, kg_score_list_2, kg_score_list_3, row_list, col_list = (
            [],
            [],
            [],
            [],
            [],
        )
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score2 = self.generate_transE_score2(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            kg_score1 = self.generate_transE_score1(
                self.all_hs[triple_index],
                self.all_ts[triple_index],
                self.all_rs[triple_index],
            )
            kg_score3 = self.generate_transE_score3(
                self.all_hs[triple_index],
                self.all_ts[triple_index],
                self.all_rs[triple_index],
            )
            # print('oooooooooooooooooooooooooooooooooooooooooooooooooooooo')
            # print(kg_score2.shape)
            # print(kg_score3.shape)
            # print('oooooooooooooooooooooooooooooooooooooooooooooooooooooo')

            kg_score_1 = kg_score1
            kg_score_2 = kg_score2
            kg_score_3 = kg_score3
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])

            kg_score_list_1.append(kg_score_1)
            kg_score_list_2.append(kg_score_2)
            kg_score_list_3.append(kg_score_3)

        kg_score1 = torch.cat(kg_score_list_1, dim=0)
        kg_score2 = torch.cat(kg_score_list_2, dim=0)
        kg_score3 = torch.cat(kg_score_list_3, dim=0)

        # print('111111111111111111111111111111111111111111111111111111111111')
        # print(kg_score1.shape)
        # print('111111111111111111111111111111111111111111111111111111111111')
        # print(kg_score2.shape)

        # print('111111111111111111111111111111111111111111111111111111111111')
        # print(kg_score3.shape)

        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax

        A_in_1 = torch.sparse.FloatTensor(indices, kg_score1, self.matrix_size).cpu()
        A_in_1 = torch.sparse.softmax(A_in_1, dim=1).to(self.device)

        A_in_2 = torch.sparse.FloatTensor(indices, kg_score2, self.matrix_size).cpu()
        A_in_2 = torch.sparse.softmax(A_in_2, dim=1).to(self.device)

        A_in_3 = torch.sparse.FloatTensor(indices, kg_score3, self.matrix_size).cpu()
        A_in_3 = torch.sparse.softmax(A_in_3, dim=1).to(self.device)

        self.A_in_1 = A_in_1
        self.A_in_2 = A_in_2
        self.A_in_3 = A_in_3

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings_1, entity_all_embeddings_1 = self.forward_1()
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward_2()
        user_all_embeddings_3, entity_all_embeddings_3 = self.forward_3()

        u_embeddings_1 = user_all_embeddings_1[user]
        i_embeddings_1 = entity_all_embeddings_1[item]
        u_embeddings_2 = user_all_embeddings_2[user]
        i_embeddings_2 = entity_all_embeddings_2[item]
        u_embeddings_3 = user_all_embeddings_3[user]
        i_embeddings_3 = entity_all_embeddings_3[item]

        scores_1 = torch.mul(u_embeddings_1, i_embeddings_1).sum(dim=1)
        scores_2 = torch.mul(u_embeddings_2, i_embeddings_2).sum(dim=1)
        scores_3 = torch.mul(u_embeddings_3, i_embeddings_3).sum(dim=1)

        scores = scores_1 + scores_2 + scores_3
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e_1, self.restore_entity_e_1 = self.forward_1()
            self.restore_user_e_2, self.restore_entity_e_2 = self.forward_2()
            self.restore_user_e_3, self.restore_entity_e_3 = self.forward_3()

        u_embeddings_1 = self.restore_user_e_1[user]
        i_embeddings_1 = self.restore_entity_e_1[: self.n_items]
        u_embeddings_2 = self.restore_user_e_2[user]
        i_embeddings_2 = self.restore_entity_e_2[: self.n_items]
        u_embeddings_3 = self.restore_user_e_3[user]
        i_embeddings_3 = self.restore_entity_e_3[: self.n_items]

        scores_1 = torch.matmul(u_embeddings_1, i_embeddings_1.transpose(0, 1))
        scores_2 = torch.matmul(u_embeddings_2, i_embeddings_2.transpose(0, 1))
        scores_3 = torch.matmul(u_embeddings_3, i_embeddings_3.transpose(0, 1))

        scores = scores_1 + scores_2 + scores_3

        return scores.view(-1)
