"""Single source of truth for the delivered FlashKDA Provider variants."""

from __future__ import annotations

from dataclasses import dataclass


CHUNK_SIZE = 64
HEAD_DIM = 128
ASCEND950PR_BLOCK_NUM = 28


@dataclass(frozen=True)
class FlashKdaVariantPolicy:
    variant_id: str
    batch: int
    heads: int
    sequence_min: int
    sequence_max: int
    group: int
    dv_base: int
    block_num: int = ASCEND950PR_BLOCK_NUM

    def matches(self, batch: int, heads: int, sequence: int) -> bool:
        return (
            batch == self.batch
            and heads == self.heads
            and self.sequence_min <= sequence <= self.sequence_max
            and sequence % CHUNK_SIZE == 0
        )


FLASH_KDA_VARIANT_POLICIES = (
    FlashKdaVariantPolicy("k3_b1_s8192", 1, 3, 8192, 8192, 128, 16),
    FlashKdaVariantPolicy(
        "k3_b1_s32768_131072", 1, 3, 32768, 131072, 256, 16
    ),
    FlashKdaVariantPolicy(
        "k3_b4_s8192_131072", 4, 3, 8192, 131072, 64, 64
    ),
    FlashKdaVariantPolicy(
        "prefill_n96_b1_s1024_65536", 1, 96, 1024, 65536, 8, 128
    ),
    FlashKdaVariantPolicy(
        "prefill_n96_b2_s10240_65536", 2, 96, 10240, 65536, 4, 128
    ),
    FlashKdaVariantPolicy(
        "prefill_n96_b4_s1024_65536", 4, 96, 1024, 65536, 2, 128
    ),
)


def matching_policies(
    batch: int, heads: int, sequence: int
) -> tuple[FlashKdaVariantPolicy, ...]:
    return tuple(
        policy
        for policy in FLASH_KDA_VARIANT_POLICIES
        if policy.matches(batch, heads, sequence)
    )


def validate_variant_policies() -> None:
    ids = [policy.variant_id for policy in FLASH_KDA_VARIANT_POLICIES]
    if len(ids) != len(set(ids)):
        raise ValueError("FlashKDA variant ids must be unique")
    for policy in FLASH_KDA_VARIANT_POLICIES:
        if (
            policy.batch <= 0
            or policy.heads <= 0
            or policy.sequence_min <= 0
            or policy.sequence_min > policy.sequence_max
            or policy.sequence_min % CHUNK_SIZE != 0
            or policy.sequence_max % CHUNK_SIZE != 0
            or policy.group <= 0
            or policy.dv_base <= 0
            or HEAD_DIM % policy.dv_base != 0
            or policy.block_num <= 0
        ):
            raise ValueError(f"invalid FlashKDA variant policy: {policy}")
        for sequence in range(
            policy.sequence_min, policy.sequence_max + 1, CHUNK_SIZE
        ):
            matches = matching_policies(policy.batch, policy.heads, sequence)
            if matches != (policy,):
                raise ValueError(
                    f"FlashKDA delivery point B{policy.batch}/N{policy.heads}/"
                    f"S{sequence} matches {[item.variant_id for item in matches]}"
                )


validate_variant_policies()
