import numpy as np
try:
    from pysmore.utils.c_alias_method import AliasTable
except:
    from utils.c_alias_method import AliasTable


class SMOReSampler:

    def __init__(self, user_list, weight_list, num_neg, num_user, num_item):
        self.num_user = num_user
        self.num_item = num_item
        self.user_list = user_list
        self.weight_list = weight_list
        self.num_neg = num_neg

        self.contexts = []
        self.vertex_sampler, self.vertex_uniform_sampler = AliasTable(), AliasTable()

        self.context_sampler, self.context_uniform_sampler = AliasTable(), AliasTable()
        self.negative_sampler = AliasTable()

        self.build_sampler()

    def triplet_generator(self):

        while True:
            u = self.draw_vertex()
            i = self.draw_context(u)

            js = []
            while len(js) < self.num_neg:
                j = self.draw_context_uniform()
                if j == i:
                    continue
                js.append(j)

            yield u, i, js

    def triplet_generator2(self):

        current = 0
        while current < self.num_user:
            u = self.draw_vertex()
            i = self.draw_context(u)

            js = []
            while len(js) < self.num_neg:
                j = self.draw_context_uniform()
                if j == i:
                    continue
                js.append(j)

            yield u, i, js
            current += 1
            if current == self.num_user:
                break

    def build_sampler(self):

        # print('Build VC-Sampler')

        vertex_distribution = [0.] * self.num_user
        vertex_uniform_distribution = [0.] * self.num_user
        context_uniform_distribution = [0.] * self.num_item
        negative_distribution = [0.] * self.num_item
        context_distribution = []

        for u in range(len(self.user_list)):
            # if len(self.user_list[u]) == 0 :
            # print(u," missing")
            # continue

            context_distribution.clear()
            for item, weight in zip(self.user_list[u], self.weight_list[u]):
                assert len(self.user_list[u]) == len(
                    self.weight_list[u]), 'item_list & rate_list not match'

                vertex_distribution[u] += weight
                negative_distribution[item] += weight
                context_distribution.append(weight)
                vertex_uniform_distribution[u] = 1.0
                context_uniform_distribution[item] = 1.0

                # accumlate the vertex be connected with
                self.contexts.append(item)

            # end-for with iterate rate & item
            self.context_sampler.append(context_distribution, 1.0)

        # print('\tCreate vertex sampler')
        self.vertex_sampler.append(vertex_distribution, 1.0)
        # print('\tCreate vertex sampler done')

        # print('\tCreate vertex uniform sampler')
        self.vertex_uniform_sampler.append(vertex_uniform_distribution, 1.0)
        # print('\tCreate vertex uniform sampler done')

        # print('\tCreate context uniform sampler')
        self.context_uniform_sampler.append(context_uniform_distribution, 1.0)
        # print('\tCreate context uniform sampler done')

        # print('\tCreate negative sampler')
        self.negative_sampler.append(negative_distribution, 0.75)
        # print('\tCreate negative sampler done') #end-for with iterate user
        # print('Build VC-Sampler done')

    def draw_vertex(self):
        return self.vertex_sampler.draw()

    def draw_context(self, v_id):
        return self.contexts[self.context_sampler.draw_by_given(v_id)]

    def draw_context_uniform(self):
        return self.context_uniform_sampler.draw()
