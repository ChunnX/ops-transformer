# CatCCOS UDMA overlay

The build uses the official CatCCOS repository at revision
`878c2e0a504be8c49c03b5e1157a7c714a628e29`.

The headers under `include/catccos` are the minimal all-to-all-v UDMA delta
required by this operator and are placed before the official CatCCOS include
directory. They preserve the validated UDMA peer-serial scheduling and
layout-aware two-dimensional copy path while all other CatCCOS, CATLASS, and
SHMEM sources continue to come from the official third-party checkout.
