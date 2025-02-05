import os
from typing import List
import numpy as np
import torch
import random
from tqdm import tqdm


class Dataloader:
    def __init__(self, train_pos, train_neg, kg_lines, train_batch_size: int = 128, neg_rate: float = 2):
        self.kg, self.rel_dict, self.n_entity = self._convert_kg(kg_lines)  # 生成实体和关系的映射字典
        self.train_pos, self.train_neg = train_pos, train_neg
        self.n_user = max(list(set(self.train_pos[:, 0]) | set(self.train_neg[:, 0]))) + 1
        self.n_item = max(list(set(self.train_pos[:, 1]) | set(self.train_neg[:, 1]))) + 1
        self._load_ratings()  # 将用户的ID从 train_pos 和 train_neg 数据中加上实体数目
        self.known_neg_dict = []  # Save the currently known negative samples
        self._add_recsys_to_kg()  # 将推荐系统的交互数据添加到知识图谱中，表示用户和项目之间的反馈（如兴趣）
        self.train_batch_size = train_batch_size  # 构建训练批次数据，包括正样本和负样本。
        self.neg_rate = neg_rate
        self.ent_num = self.n_entity + self.n_user
        self.rel_num = len(self.rel_dict)

    def _add_recsys_to_kg(self):
        # Add the interaction data to the kg as the extra relation
        self.rel_dict['feedback_recsys'] = max([self.rel_dict[key] for key in self.rel_dict]) + 1
        for interaction in self.train_pos:
            self.kg.append((interaction[0], self.rel_dict['feedback_recsys'], interaction[1]))
        for interaction in self.train_neg:
            self.known_neg_dict.append((interaction[0], self.rel_dict['feedback_recsys'], interaction[1]))

    def _load_ratings(self):
        # Loading known interaction data
        # 防止实体和item编号冲突
        self.n_entity = max(self.n_item, self.n_entity)
        for i in range(len(self.train_pos)):
            self.train_pos[i][0] += self.n_entity
        for i in range(len(self.train_neg)):
            self.train_neg[i][0] += self.n_entity

    def _convert_kg(self, lines):  # 返回经过处理的知识图谱和最大实体编号。
        # Load the kg data and convert the relation type to int
        entity_set = set()
        kg = []
        rel_dict = {}
        for line in open('./relation2id.txt', encoding='utf8').readlines():
            elements = line.replace('\n', '').split('\t')
            rel_dict[elements[0]] = int(elements[1])
        for line in lines:
            array = line.strip().split('\t')
            head = int(array[0])
            relation = rel_dict[array[1]]
            tail = int(array[2])
            kg.append((head, relation, tail))
            entity_set.add(head)
            entity_set.add(tail)

        print('number of entities (containing items): %d' % len(entity_set))
        print('number of relations: %d' % len(rel_dict))
        return kg, rel_dict, max(list(entity_set)) + 1 if len(entity_set) > 0 else 0

    def get_user_pos_item_list(self):  # 返回每个用户的正向交互项目列表。
        # Get the known positive items for each user
        train_user_pos_item = {}
        all_record = np.concatenate([self.train_pos, self.train_neg], axis=0)
        for record in self.train_pos:
            user, item = record[0] - self.n_entity, record[1]
            if user not in train_user_pos_item:
                train_user_pos_item[user] = set()
            train_user_pos_item[user].add(item)
        item_list = list(set(all_record[:, 1]))
        return item_list, train_user_pos_item

    def get_training_batch(self):  # 构建训练批次数据，包括正样本和负样本。
        pos_data = [fact for fact in self.kg]
        neg_data = [fact for fact in self.known_neg_dict]
        hr_tail_set = {}
        rt_head_set = {}
        for fact in pos_data + neg_data:
            if (fact[0], fact[1]) not in hr_tail_set:
                hr_tail_set[(fact[0], fact[1])] = set()
            if (fact[1], fact[2]) not in rt_head_set:
                rt_head_set[(fact[1], fact[2])] = set()
            hr_tail_set[(fact[0], fact[1])].add(fact[2])
            rt_head_set[(fact[1], fact[2])].add(fact[0])
        sample_failed_time = 0
        sample_failed_max = len(self.kg) * self.neg_rate
        # Sample extra negative training samples
        while len(neg_data) < len(self.kg) * self.neg_rate and sample_failed_time < sample_failed_max:
            # Set sample try times upper bound.
            if sample_failed_time < sample_failed_max:
                for fact in self.kg:
                    if len(neg_data) >= len(self.kg) * self.neg_rate:
                        break
                    if random.random() > 0.5:
                        if fact[0] >= self.n_entity:
                            tail = random.randint(0, self.n_item - 1)
                            while tail in hr_tail_set[(fact[0], fact[1])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                tail = random.randint(0, self.n_item - 1)
                        else:
                            tail = random.randint(0, self.n_entity - 1)
                            while tail in hr_tail_set[(fact[0], fact[1])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                tail = random.randint(0, self.n_entity - 1)
                        if sample_failed_time < sample_failed_max:
                            hr_tail_set[(fact[0], fact[1])].add(tail)
                            neg_data.append((fact[0], fact[1], tail))
                    else:  # consider whether the head entity is a user
                        if fact[0] >= self.n_entity:
                            head = random.randint(self.n_entity, self.n_entity + self.n_user - 1)
                            while head in rt_head_set[(fact[1], fact[2])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                head = random.randint(self.n_entity, self.n_entity + self.n_user - 1)
                        else:
                            head = random.randint(0, self.n_entity - 1)
                            while head in rt_head_set[(fact[1], fact[2])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                head = random.randint(0, self.n_entity - 1)
                        if sample_failed_time < sample_failed_max:
                            rt_head_set[(fact[1], fact[2])].add(head)
                            neg_data.append((head, fact[1], fact[2]))
        random.shuffle(pos_data)
        random.shuffle(neg_data)
        pos_batches = np.array_split(pos_data, max(1, len(pos_data) // self.train_batch_size))
        neg_batches = np.array_split(neg_data, len(pos_batches))
        pos_batches = [batch.transpose() for batch in pos_batches]
        neg_batches = [batch.transpose() for batch in neg_batches]
        return [[pos_batches[index], neg_batches[index]] for index in range(len(pos_batches))]


class TransE(torch.nn.Module):
    def __init__(self, ent_num: int, rel_num: int, dataloader: Dataloader, dim: int = 100, l1: bool = True,
                 margin: float = 1, learning_rate: float = 0.05, weight_decay: float = 1e-4, device_index: int = 0):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(device_index)) if device_index >= 0 else torch.device('cpu')
        self.ent_num: int = ent_num
        self.rel_num: int = rel_num
        self.dataloader = dataloader
        self.dim: int = dim
        self.l1: bool = l1
        self.margin: float = margin
        self.learning_rate: float = learning_rate
        self.weight_decay = weight_decay
        self.ent_embedding = torch.nn.Embedding(self.ent_num, self.dim, device=self.device)
        self.rel_embedding = torch.nn.Embedding(self.rel_num, self.dim, device=self.device)

    def forward(self, head, rel, tail) -> torch.Tensor:
        """
        Compute the similarity score between the head entity, relation, and tail entity.
        Args:
            head (torch.Tensor): The indices of the head entities.
            rel (torch.Tensor): The indices of the relations.
            tail (torch.Tensor): The indices of the tail entities.
        Returns:
            torch.Tensor: The negative similarity score (distance) between the head entity,
                          relation, and tail entity. The score is computed based on either
                          L1 (Manhattan) or L2 (Euclidean) distance.
        """
        head_emb = self.ent_embedding(torch.IntTensor(head).to(self.device))
        tail_emb = self.ent_embedding(torch.IntTensor(tail).to(self.device))
        rel_emb = self.rel_embedding(torch.IntTensor(rel).to(self.device))
        if self.l1:
            score = torch.sum(torch.abs(torch.subtract(torch.add(head_emb, rel_emb), tail_emb)), dim=-1, keepdim=True)
        else:
            score = torch.sum(torch.square(torch.subtract(torch.add(head_emb, rel_emb), tail_emb)), dim=-1,
                              keepdim=True)
        return -score

    def optimize(self, pos, neg):
        """
        Calculate the Margin Loss between positive and negative samples and update the model.
        Args:
            pos (tuple): A tuple containing positive samples, where pos[0], pos[1], and pos[2]
                         represent the head, relation, and tail of positive triplets respectively.
            neg (tuple): A tuple containing negative samples, where neg[0], neg[1], and neg[2]
                         represent the head, relation, and tail of negative triplets respectively.
        Returns:
            torch.Tensor: The computed Margin Loss, representing the difference between positive
                          and negative scores, ensuring that the positive samples are closer
                          to the correct entity relations than negative samples.
        """
        pos_score = self.forward(pos[0], pos[1], pos[2])
        neg_score = self.forward(neg[0], neg[1], neg[2])

        pos_matrix = torch.matmul(pos_score, torch.t(torch.ones_like(neg_score)))
        neg_matrix = torch.t(torch.matmul(neg_score, torch.t(torch.ones_like(pos_score))))
        loss = torch.mean(torch.clamp(torch.add(torch.subtract(neg_matrix, pos_matrix), self.margin), min=0))
        return loss

    def ctr_eval(self, eval_batches: List[np.array]):
        """
        Evaluate the CTR (Click-Through Rate) task, predicting the interest level of each user
        for the items in the given test data.
        Args:
            eval_batches (List[np.array]): A list of batches containing user-item interaction data.
                                           Each batch is represented as a NumPy array, where:
                                           - batch[0] represents the user IDs,
                                           - batch[1] represents the item IDs.
        Returns:
            np.array: An array of predicted scores for the interest level (CTR) of each user-item pair.
                      These scores indicate how likely a user is to interact with a particular item.
        """
        eval_batches = [batch.transpose() for batch in eval_batches]
        scores = []
        for batch in eval_batches:
            rel = [self.dataloader.rel_dict['feedback_recsys'] for _ in range(len(batch[0]))]
            # User ID in the mixed KG should add the number of the entities in the origin KG
            # print("self.dataloader.ent_num = ", self.dataloader.ent_num)
            # print("self.ent_num:", self.ent_num)
            # print("batch[0]:", batch[0])
            # print("max in batch[0]:", max(batch[0]))
            # print("min in batch[0]:", min(batch[0]))
            # print("batch[1]:", batch[1])
            # print("max in batch[1]:", max(batch[1]))
            # print("min in batch[1]:", min(batch[1]))
            # print("self.dataloader.n_entity:",  self.dataloader.n_entity)
            # print("self.dataloader.n_user:",  self.dataloader.n_user)
            # print("batch[0] + self.dataloader.n_user = ", batch[0] + self.dataloader.n_user - 1)
            # print("max in (batch[0] + self.dataloader.n_user) = ", max(batch[0] + self.dataloader.n_user))

            # score = torch.squeeze(self.forward(batch[0] + self.dataloader.n_entity, rel, batch[1]), dim=-1)
            score = torch.squeeze(self.forward(batch[0] + self.dataloader.n_user, rel, batch[1]), dim=-1)
            scores.append(score.cpu().detach().numpy())
        scores = np.concatenate(scores, axis=0)
        return scores

    def top_k_eval(self, users: List[int], k: int = 5):
        """
        Evaluate the top-k recommendation task, returning the top k most interesting items
        for each user based on the predicted scores from the model.
        Args:
            users (List[int]): A list of user IDs for which to generate the top-k recommendations.
            k (int): The number of top items to return for each user. Default is 5.
        Returns:
            List[List[int]]: A list of lists, where each inner list contains the top k recommended
                             items for a particular user, sorted by their predicted interest scores.
        """
        # Get the known positive items for each user
        item_list, train_user_pos_item = self.dataloader.get_user_pos_item_list()
        sorted_list = []
        for user in users:
            # 防止head重复
            head = [user + self.dataloader.n_user for _ in range(len(item_list))]
            rel = [self.dataloader.rel_dict['feedback_recsys'] for _ in range(len(item_list))]
            tail = item_list

            # print("max in head:", max(head))
            scores = torch.squeeze(self.forward(head, rel, tail), dim=-1)
            score_ast = np.argsort(scores.cpu().detach().numpy(), axis=-1)[::-1]
            sorted_items = []
            for index in score_ast:
                if len(sorted_items) >= k:
                    break
                # 结果不包含已知训练数据
                if user not in train_user_pos_item or item_list[index] not in train_user_pos_item[user]:
                    sorted_items.append(item_list[index])
            sorted_list.append(sorted_items)
        return sorted_list

    def train_TransE(self, epoch_num: int, output_log=False):
        """
        Args:
            epoch_num (int): The number of epochs to train the model.
            output_log (bool): If True, the method will print the average loss after each epoch.
                               Default is False.
        """
        # Use Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in tqdm(range(epoch_num)):
            train_batches = self.dataloader.get_training_batch()
            losses = []
            for batch in train_batches:
                loss = self.optimize(batch[0], batch[1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())
            if output_log:
                print("The loss after the", epoch, "epochs is", np.mean(losses))


class KGRS:
    def __init__(self, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        """
        Initialize the Algorithm
        :param train_pos: The Positive Samples in the Training Set, is a numpy matrix with shape (n,3),
                          while `n` is the number of positive samples, and in each sample, the first
                          number represent the user, the second represent the item, and the last indicate
                          interest or not. e.g. [[1,2,1], [2,5,1],[1,3,1]] indicate that user 1 has
                          interest in item 2 and item 3, user 2 has interest in item 5.
        :param train_neg: The Negative Samples in the Training Set, is a numpy matrix with shape (n,3),
                          while `n` is the number of positive samples, and in each sample, the first
                          number represent the user, the second represent the item, and the last indicate
                           interest or not. e.g. [[1,4,0], [2,2,0],[1,5,0]] indicate that user 1 has no
                           interest in item 4 and item 5, user 2 has no interest in item 2.
        :param kg_lines: The Knowledge Graph Lines, is a list of strings. Each element in the list is a
                         string representing one relation in the Knowledge Graph. The string can be split
                         into 3 parts by '\t', the first part is head entity, the second part is relation
                         type, and the third part is tail entity. E.g. ["749\tfilm.film.writer\t2347"]
                         represent a Knowledge Graph only has one relation, in that relation, head entity
                         is 749, tail entity is 2347, and the relation type is "film.film.writer".
        """
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        config = {"batch_size": 256, "eval_batch_size": 1024, "neg_rate": 2, "emb_dim": 128, "l1": False, "margin": 15,
                  "learning_rate": 0.005, "weight_decay": 0.003, "epoch_num": 40}
        self.batch_size = config["batch_size"]
        self.eval_batch_size = config["eval_batch_size"]
        self.neg_rate = config["neg_rate"]
        self.emb_dim = config["emb_dim"]
        self.l1 = config["l1"]
        self.margin = config["margin"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epoch_num = config["epoch_num"]
        self.device_index = -1
        self.kg = kg_lines
        self.dataloader = Dataloader(train_pos, train_neg, self.kg, neg_rate=self.neg_rate,
                                     train_batch_size=self.batch_size)
        self.model = TransE(ent_num=self.dataloader.ent_num, rel_num=self.dataloader.rel_num,
                            dataloader=self.dataloader, margin=self.margin, dim=self.emb_dim, l1=self.l1,
                            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                            device_index=self.device_index)

    def training(self):
        """
        Train the Recommendation System
        :return: None
        """
        self.model.train_TransE(epoch_num=self.epoch_num)

    def eval_ctr(self, test_data: np.array) -> np.array:
        """
        Evaluate the CTR Task result
        :param test_data: The test data that you need to predict. The data is a numpy matrix with shape (n, 2),
                          while `n` is the number of the test samples, and in each sample, the first dimension
                          is the user and the second is the item. e.g. [[2, 4], [2, 6], [4, 1]] means you need
                          to predict the interest level of: from user 2 to item 4, from user 2 to item 6, and
                          from user 4 to item 1.
        :return: The prediction result, is an n dimension numpy array, and the i-th dimension means the predicted
                 interest level of the i-th sample, while the higher score means user has higher interest in the
                 item. e.g. while test_data=[[2, 4], [2, 6], [4, 1]], the return value [1.2, 3.3, 0.7] means that
                 the interest level from user 2 to item 6 is highest in these samples, and interest level from
                 user 2 to item 4 is second highest, interest level from user 4 to item 1 is lowest.
        """
        eval_batches = np.array_split(test_data, len(test_data) // self.eval_batch_size)
        return self.model.ctr_eval(eval_batches)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        """
        Evaluate the Top-K Recommendation Task result
        :param users: The list of the id of the users that need to be recommended items. e.g. [2, 4, 8] means
                      you need to recommend k items for the user 2, 4, 8 respectively, and the term of the user
                      and recommended item cannot have appeared in the train_pos data.
        :param k: The number of the items recommended to each user. In this project, k=5.
        :return: The items recommended to the users respectively, and the order of the items should be sorted by
                 the interest level of the user to the item. e.g. while user=[2, 4, 8] and k=5, the return value
                 is [[2, 5, 7, 4, 6],[3, 5, 2, 1, 21],[12, 43, 7, 3, 2]] means you will recommend item 2, 5, 7,
                 4, 6 to user 2, recommend item 3, 5, 2, 1, 21 to user 4, and recommend item 12, 43, 7, 3, 2 to
                 user 8, and the interest level from user to the item in the recommend list are degressive.
        """
        return self.model.top_k_eval(users, k=k)
