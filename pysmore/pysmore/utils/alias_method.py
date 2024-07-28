from typing import List
from collections import deque
import random

class AliasTable():

    def __init__(self):
        self.offsets = deque()
        self.branchs = deque()

        self.alias_position  = deque()
        self.alias_probality = deque()


    def append(self, distribution:List, power:float) -> None:

        offset = len(self.alias_position)
        branch = len(distribution)
        self.offsets.append(offset)
        self.branchs.append(branch)

        if len(distribution) == 0:
            return

        norm_prob = deque()

        sum, norm = 0., 0.
        for weight in distribution:
            sum += weight ** power 
            self.alias_position.append(-1)
            self.alias_probality.append(1.1)

        try:
            norm = len(distribution)/sum
        except ZeroDivisionError:
            print('maybe the user list is empty')
            exit(1)


        for weight in distribution:
            norm_prob.append( weight**power*norm )

        small_block, large_block = deque(), deque()

        for pos in range(len(norm_prob)):
            if norm_prob[pos] < 1:
                small_block.append(pos)
            if norm_prob[pos] > 1:
                large_block.append(pos)

        small_pos, large_pos = -1, -1


        while len(small_block) and len(large_block):
            small_pos = small_block.popleft()
            large_pos = large_block.popleft()

            self.alias_position[offset+small_pos] = offset+large_pos
            self.alias_probality[offset+small_pos] = norm_prob[small_pos]

            norm_prob[large_pos] = norm_prob[large_pos]+norm_prob[small_pos]-1

            if norm_prob[large_pos] < 1:
                small_block.append(large_pos)
            else:
                large_block.append(large_pos)

        while len(large_block):
            large_pos = large_block.popleft()

        while len(small_block):
            large_pos = small_block.popleft()

    def draw(self):
        sam_pos = random.randrange(0, len(self.alias_position))
        # sam_pos = random.randint(0, len(self.alias_position)-1)

        sam_prob = random.uniform(0, 1)

        if sam_prob < self.alias_probality[sam_pos]:
            return sam_pos
        else:
            return self.alias_position[sam_pos]

    def draw_by_given(self, idx):

        sam_pos = self.offsets[idx]+ random.randrange(0, self.branchs[idx])

        sam_prob = random.uniform(0, 1)

        if sam_prob < self.alias_probality[sam_pos]:
            return sam_pos
        else:
            return self.alias_position[sam_pos]

