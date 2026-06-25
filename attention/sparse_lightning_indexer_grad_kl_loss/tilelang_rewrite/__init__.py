from .sparse_lightning_indexer_grad_kl_loss import (
    sparse_lightning_indexer_grad_kl_loss,
    SparseLightningIndexerGradKLLossTilingData,
)

# Plan A: K-parallel kernel + Python postprocessing
from .sparse_lightning_indexer_grad_kl_loss_planA import (
    sparse_lightning_indexer_grad_kl_loss as sparse_lightning_indexer_grad_kl_loss_planA,
)

# Plan B: Dual-kernel with Python bridge
from .sparse_lightning_indexer_grad_kl_loss_planB import (
    sparse_lightning_indexer_grad_kl_loss as sparse_lightning_indexer_grad_kl_loss_planB,
)
