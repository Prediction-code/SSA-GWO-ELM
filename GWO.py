# -*- coding: utf-8 -*-
"""
Grey Wolf Optimizer (Mirjalili et al., 2014).
连续空间搜索 + 取整，用于 ELM 各分量隐层神经元数等离散超参。
"""
import numpy as np


class GWO:
    """
    最小化 fitness(discrete_vector)，discrete_vector 为 length=dim 的整型数组。
    """

    def __init__(
        self,
        dim,
        lb,
        ub,
        fitness,
        n_wolves=30,
        max_iter=50,
        seed=None,
    ):
        self.dim = int(dim)
        self.lb = np.atleast_1d(np.asarray(lb, dtype=float))
        self.ub = np.atleast_1d(np.asarray(ub, dtype=float))
        if self.lb.size == 1:
            self.lb = np.full(self.dim, float(self.lb[0]))
        if self.ub.size == 1:
            self.ub = np.full(self.dim, float(self.ub[0]))
        if self.lb.shape[0] != self.dim or self.ub.shape[0] != self.dim:
            raise ValueError("lb/ub 长度须等于 dim")
        self.fitness = fitness
        self.n_wolves = max(3, int(n_wolves))
        self.max_iter = int(max_iter)
        self.rng = np.random.default_rng(seed)
        self.gbest = None
        self.gbest_score = np.inf

    def _discrete(self, x):
        xi = np.rint(np.clip(x, self.lb, self.ub)).astype(int)
        return np.maximum(xi, 1)

    def optimize(self):
        n, dim = self.n_wolves, self.dim
        pos = self.rng.uniform(self.lb, self.ub, size=(n, dim))

        def eval_pos(p):
            d = self._discrete(p)
            return float(self.fitness(d)), d

        scores = np.zeros(n)
        discs = np.zeros((n, dim), dtype=int)
        for i in range(n):
            scores[i], discs[i] = eval_pos(pos[i])

        order = np.argsort(scores)
        self.gbest_score = float(scores[order[0]])
        self.gbest = discs[order[0]].copy()

        for t in range(self.max_iter):
            order = np.argsort(scores)
            alpha_pos = pos[order[0]].copy()
            beta_pos = pos[order[1]].copy()
            delta_pos = pos[order[2]].copy()

            a = 2.0 - t * (2.0 / max(self.max_iter, 1))

            for i in range(n):
                for j in range(dim):
                    r1, r2 = self.rng.random(), self.rng.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - pos[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = self.rng.random(), self.rng.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - pos[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = self.rng.random(), self.rng.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - pos[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    pos[i, j] = (X1 + X2 + X3) / 3.0
                    pos[i, j] = np.clip(pos[i, j], self.lb[j], self.ub[j])

            for i in range(n):
                scores[i], discs[i] = eval_pos(pos[i])
                if scores[i] < self.gbest_score:
                    self.gbest_score = float(scores[i])
                    self.gbest = discs[i].copy()

        return self
