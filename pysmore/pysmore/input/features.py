from typing import Optional
from dataclasses import dataclass

@dataclass
class SequenceFeature(object):
    vocab_size: int
    name: Optional[str]
    embed_dim: Optional[int] = None
    pooling: Optional[str] = 'mean'
    shared_with: Optional[str] = None
    padding_idx: Optional[int] = None

    def __post_init__(self):
        if self.embed_dim == 'auto':
            self.embed_dim = 6 * int(pow(self.vocab_size, 0.25))


@dataclass
class SparseFeature:
    vocab_size: int
    name: Optional[str] = None
    embed_dim: Optional[int] = None
    shared_with: Optional[str] = None

    def __post_init__(self):
        if self.embed_dim == 'auto':  
            self.embed_dim = 6 * int(pow(self.vocab_size, 0.25))

@dataclass
class DenseFeature:
    name: Optional[str] = None,
    embed_dim: Optional[int] = 1
    shared_with: Optional[str] = None

    def __post_init__(self):
        if self.embed_dim == 'auto': 
            self.embed_dim = 1

if __name__ == '__main__':
    a = DenseFeature()
    b = SparseFeature(4)
    print(a)
    print(b)
    

    
