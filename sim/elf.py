# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Extract MIX AIC/AIV images using symbols from the one-key linked ELF."""

import struct


def images(path, tiling_key=None):
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a 64-bit little-endian Ascend ELF")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, names_idx = struct.unpack_from("<HHH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + j * size) for j in range(count)
    ]
    ns = sections[names_idx]
    names = data[ns[4] : ns[4] + ns[5]]
    named = {names[s[0] :].split(b"\0")[0].decode(): s for s in sections}
    if ".text" not in named or ".symtab" not in named:
        raise ValueError("ELF lacks .text or .symtab")
    allocated = [s for s in sections if s[2] & 2 and s[5]]
    # CANN keeps .rela.text even in its final ET_EXEC. Those relocations have
    # already been applied by the device linker; intermediate ET_REL is unsafe.
    if struct.unpack_from("<H", data, 16)[0] != 2:
        raise ValueError("Need the final linked ET_EXEC, not an intermediate ET_REL .o")
    end = max(s[3] + s[5] for s in allocated)
    if end > 128 << 20:
        raise ValueError("Unexpected sparse ELF address layout")
    image = bytearray(end)
    for s in allocated:
        if s[1] != 8:  # SHT_NOBITS is zero initialized.
            image[s[3] : s[3] + s[5]] = data[s[4] : s[4] + s[5]]
    syms = named[".symtab"]
    strings_section = sections[syms[6]]
    strings = data[strings_section[4] : strings_section[4] + strings_section[5]]
    result, symbols = {}, {}
    for pos in range(syms[4], syms[4] + syms[5], syms[9]):
        name, info, _, section, value, length = struct.unpack_from("<IBBHQQ", data, pos)
        name = strings[name:].split(b"\0")[0].decode()
        for kind in ("aic", "aiv"):
            if name.endswith("_mix_" + kind) and section and info & 15 == 2:
                if kind in result or value >= end or length < 256:
                    raise ValueError("Ambiguous or empty MIX entry symbol")
                if tiling_key is not None and not name.endswith(
                    f"_{tiling_key}_mix_{kind}"
                ):
                    raise ValueError(
                        f"MIX entry does not match requested key {tiling_key}: {name}"
                    )
                result[kind] = bytes(image[value:])
                symbols[kind] = {
                    "name": name,
                    "offset": value,
                    "symbol_bytes": length,
                    "image_bytes": len(result[kind]),
                }
    if set(result) != {"aic", "aiv"}:
        raise ValueError(
            "ELF must contain exactly one AIC and one AIV entry; check dtype and key"
        )
    return result, symbols
