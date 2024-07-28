import cython
import random
import time

from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free, srand
from libc.time cimport time

"""
cdef int uniform_distribution(int start, int stop) :
    srand(time(NULL))

    # int range = stop - start + 1;
    cdef int range = stop - start
    cdef int rand_num = rand() % range

    return rand_num + start

"""
cdef int uniform_distribution(int rangeLow, int rangeHigh) :
    # cdef double myRand = rand()/(1.0 + RAND_MAX); 
    srand(time(NULL))
    # cdef int range = rangeHigh - rangeLow + 1;
    cdef int range = rangeHigh - rangeLow;
    return int(rand() * range) + rangeLow;

cdef int c_randrange(int a, int b):
    return a + int((b-a)*rand()/(RAND_MAX+1.0))


cdef class AliasTable:

    # cdef:
        # int* offsets
        # int* branchs
        # int* alias_position
        # float* alias_probality

    # cdef float *my_array
    cdef int* offsets
    cdef int* branchs
    cdef float* alias_probality
    cdef int* alias_position

    cdef int offsets_idx 
    cdef int branchs_idx
    cdef int alias_probality_idx
    cdef int alias_position_idx


    def __cinit__(self):
        # allocate some memory (uninitialised, may contain arbitrary data)
        self.offsets = <int*> malloc(
            100000000 * sizeof(int))
        self.branchs = <int*> malloc(
            100000000 * sizeof(int))
        self.alias_position = <int*> malloc(
            100000000 * sizeof(int))
        self.alias_probality = <float*> malloc(
            100000000 * sizeof(float))

        self.offsets_idx = 0 
        self.branchs_idx = 0
        self.alias_probality_idx = 0
        self.alias_position_idx = 0

        # if not self.offsets:
            # raise MemoryError()


    def append(self, distribution, float power):
        # Convert list to array
        cdef size_t n = len(distribution)
        if n == 0:
            return
        # cdef float *data = <float *> malloc(n * sizeof(float))
        cdef unsigned int i
        # done.

        cdef float *norm_prob = <float*> malloc( n * sizeof(float))
        
        cdef float sum = 0.
        cdef float norm = 0.

        cdef float small_pos = -1
        cdef float large_pos = -1

        cdef int small_block_id = -1
        cdef int large_block_id = -1
        cdef float *small_block = <float*> malloc( n * sizeof(float))
        cdef float *large_block = <float*> malloc( n * sizeof(float))


        cdef int t_idx = 0

        cdef long offset = self.alias_position_idx
        cdef long branch = n

        self.offsets[self.offsets_idx] = offset
        self.branchs[self.branchs_idx] = branch
        self.offsets_idx += 1
        self.branchs_idx += 1

        for i in range(n):
            sum += pow(distribution[i], power)
            self.alias_position[self.alias_position_idx] = offset
            self.alias_probality[self.alias_probality_idx] = branch
            self.alias_position_idx += 1
            self.alias_probality_idx += 1

        norm = n/sum

        for i in range(n):
            norm_prob[i] = pow(distribution[i],power)
            if norm_prob[i] < 1:
                small_block_id+=1
                small_block[small_block_id] = i

            if norm_prob[i] > 1:
                large_block_id+=1
                large_block[large_block_id] = i

        while small_block_id > 0 and large_block_id > 0:
            small_pos = small_block[small_block_id]
            small_block_id -= 1
        
            large_pos = large_block[large_block_id]
            large_block_id -= 1

            t_idx = <int>(offset+small_pos)
            self.alias_position[t_idx] = <int>(offset + large_pos)
            self.alias_probality[t_idx] = norm_prob[(<int>small_pos)]

            if norm_prob[<int>large_pos] < 1:
                small_block_id+=1
                small_block[small_block_id] = large_pos
            else:
                large_block_id+=1
                large_block[large_block_id] = large_pos

        while large_block_id > 0:
            large_pos = large_block[large_block_id]
            large_block_id -= 1

        while small_block_id > 0:
            small_pos = small_block[small_block_id]
            small_block_id -= 1


    def draw(self):
        sam_pos = int(random.uniform(0, self.alias_position_idx))
        # sam_pos = uniform_distribution(0, self.alias_position_idx)

        # sam_prob = random.uniform(0, 1)
        sam_prob = int(random.uniform(0, 1))
        # sam_prob = uniform_distribution(0, 1)

        if sam_prob < self.alias_probality[sam_pos]:
            return sam_pos
        else:
            return self.alias_position[sam_pos]

    def draw_by_given(self, idx):
        # sam_pos = self.offsets[idx] + uniform_distribution(0, self.branchs[idx])
        # sam_pos = self.offsets[idx] + randrange(0, self.branchs[idx])
        sam_pos = self.offsets[idx] + int(random.uniform(0, self.branchs[idx]))


        sam_prob = int(random.uniform(0, 1))
        # sam_prob = c_randrange(0, 1)

        if sam_prob < self.alias_probality[sam_pos]:
            return sam_pos
        else:
            return self.alias_position[sam_pos]

    def draw_by_given_safely(self, idx):
        if self.branchs[idx] == 0:
            return -1
        sam_pos = random.randrange(0, self.alias_position_idx)
        # sam_pos = uniform_distribution(0, self.alias_position_idx)

        sam_prob = random.uniform(0, 1)
        # sam_prob = uniform_distribution(0, 1)

        if sam_prob < self.alias_probality[sam_pos]:
            return sam_pos
        else:
            return self.alias_position[sam_pos]
