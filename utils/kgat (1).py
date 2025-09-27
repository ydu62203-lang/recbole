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
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c
from abc import ABC, abstractmethod
from typing import Tuple


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
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
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
        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = self.layers[-1] * 2
        print("inner size:", inner_size)
        self.projection_head.append(
            torch.nn.Linear(inner_size, inner_size * 4, bias=False)
        )
        self.projection_head.append(
            torch.nn.BatchNorm1d(inner_size * 4, eps=1e-12, affine=affine)
        )
        self.projection_head.append(
            torch.nn.Linear(inner_size * 4, inner_size, bias=False)
        )
        self.projection_head.append(
            torch.nn.BatchNorm1d(inner_size, eps=1e-12, affine=affine)
        )
        self.mode = 0

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
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
        self.entity = nn.Embedding(self.n_entities, self.rank)
        self.rel = nn.Embedding(self.n_relations, self.rank)
        self.bh = nn.Embedding(self.n_entities, 1)
        self.bh.weight.data = torch.zeros((self.n_entities, 1), dtype=self.data_type)
        self.bt = nn.Embedding(self.n_entities, 1)
        self.bt.weight.data = torch.zeros((self.n_entities, 1), dtype=self.data_type)

        self.entity.weight.data = self.init_size * torch.randn(
            (self.sizes[0], self.rank), dtype=self.data_type
        )
        self.rel.weight.data = self.init_size * torch.randn(
            (self.sizes[1], 2 * self.rank), dtype=self.data_type
        )
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = (
            2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        )
        # self.multi_c = args.multi_c
        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)

        self.c = nn.Parameter(c_init, requires_grad=True)

        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = (
            2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        )
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn(
            (self.sizes[1], self.rank), dtype=self.data_type
        )
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1.0 / np.sqrt(self.rank)]).double()

        self.regularizer = N3  # 未写完

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
                dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True)
                .adjacency_matrix(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

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

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

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

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()
        kgat_all_embeddings = torch.cat((user_all_embeddings, entity_all_embeddings), 0)

        user_all_embeddings_1, entity_all_embeddings_1 = self.forward_1()
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward_2()

        user_rand_samples = self.rand_sample(
            user_all_embeddings_1.shape[0], size=user.shape[0] // 8, replace=False
        )
        entity_rand_samples = self.rand_sample(
            entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False
        )

        cts_embedding_1 = user_all_embeddings_1[torch.tensor(user_rand_samples)]
        cts_embedding_2 = user_all_embeddings_2[torch.tensor(user_rand_samples)]

        e_cts_embedding_1 = entity_all_embeddings_1[torch.tensor(entity_rand_samples)]
        e_cts_embedding_2 = entity_all_embeddings_2[torch.tensor(entity_rand_samples)]

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        cts_embedding_1 = self.projection_head_map(cts_embedding_1, self.mode)
        cts_embedding_2 = self.projection_head_map(cts_embedding_2, 1 - self.mode)
        e_cts_embedding_1 = self.projection_head_map(e_cts_embedding_1, self.mode)
        e_cts_embedding_2 = self.projection_head_map(e_cts_embedding_2, 1 - self.mode)

        u_embeddings = self.projection_head_map(u_embeddings, self.mode)
        pos_embeddings = self.projection_head_map(pos_embeddings, 1 - self.mode)

        self.mode = 1 - self.mode

        cts_loss = self.cts_loss(
            cts_embedding_1,
            cts_embedding_2,
            temp=1.0,
            batch_size=cts_embedding_1.shape[0],
        )

        e_cts_loss = self.cts_loss(
            e_cts_embedding_1,
            e_cts_embedding_2,
            temp=1.0,
            batch_size=e_cts_embedding_1.shape[0],
        )

        ui_cts_loss = self.cts_loss(
            u_embeddings, pos_embeddings, temp=1.0, batch_size=u_embeddings.shape[0]
        )

        #        cts_loss_1 = self.cts_loss(cts_embedding, cts_embedding_1, temp=0.1,
        #                                                        batch_size=cts_embedding_1.shape[0])
        #        cts_loss_2 = self.cts_loss(cts_embedding, cts_embedding_2, temp=0.1,
        #                                                        batch_size=cts_embedding_1.shape[0])

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        #        print("cts_loss:", cts_loss, e_cts_loss, ui_cts_loss)
        loss = (
            mf_loss
            + self.reg_weight * reg_loss
            + 0.01 * (cts_loss + e_cts_loss + ui_cts_loss)
        )
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

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return -hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""

        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

    def score(self, lhs, rhs):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == "constant":
            return self.gamma.item() + score
        elif self.bias == "learn":
            return lhs_biases + rhs_biases + score
        else:
            return score

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return -hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
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

    def calculate_kg_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        input_batch = torch.stack((h, r, pos_t), dim=1)
        neg_samples = torch.stack((h, r, neg_t), dim=1)

        loss = self.calculate_loss1(input_batch, neg_samples)

        return loss

    def generate_transE_score(self, hs, ts, r):
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

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        drop_edge_1 = self.rand_sample(
            indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False
        )
        indices_1 = indices.view(-1, 2)[torch.tensor(drop_edge_1)].view(2, -1)
        kg_score_1 = kg_score[torch.tensor(drop_edge_1)]
        A_in_1 = torch.sparse.FloatTensor(indices_1, kg_score_1, self.matrix_size).cpu()
        A_in_1 = torch.sparse.softmax(A_in_1, dim=1).to(self.device)

        drop_edge_2 = self.rand_sample(
            indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False
        )
        indices_2 = indices.view(-1, 2)[torch.tensor(drop_edge_2)].view(2, -1)
        kg_score_2 = kg_score[torch.tensor(drop_edge_2)]
        A_in_2 = torch.sparse.FloatTensor(indices_2, kg_score_2, self.matrix_size).cpu()
        A_in_2 = torch.sparse.softmax(A_in_2, dim=1).to(self.device)

        self.A_in = A_in
        self.A_in_1 = A_in_1
        self.A_in_2 = A_in_2

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[: self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
