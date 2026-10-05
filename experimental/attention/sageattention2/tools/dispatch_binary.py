#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
"""
dispatch_binary.py: CANN ACLNN 静态二进制调度生成器

负责将毕昇编译器单次编译出的 Ascend C 设备核目标文件 (sageattention2_bisheng.o)，
分发并映射为 CANN ACLNN 静态注册表所需的 16 个 static binary targets:
  - 自动解包 .aicore_binary 纯设备 ELF (e_machine 0x1029, ET_EXEC)
  - 对 16 个 DataType/Format 静态档位，逐一生成同时支持 FP16 (tilingKey: 1) 与
    BF16 (tilingKey: 2) 的混核符号与元数据 section:
      SageAttention2_<hash>_1_mix_aic / SageAttention2_<hash>_2_mix_aic
      SageAttention2_<hash>_1_mix_aiv / SageAttention2_<hash>_2_mix_aiv
      SageAttention2_<hash>_1_mix_aic__ / SageAttention2_<hash>_2_mix_aic__
      SageAttention2_<hash>_1_mix_aiv__ / SageAttention2_<hash>_2_mix_aiv__
      .ascend.meta.SageAttention2_<hash>_1_mix_aic / .ascend.meta.SageAttention2_<hash>_2_mix_aic
      .ascend.meta.SageAttention2_<hash>_1_mix_aiv / .ascend.meta.SageAttention2_<hash>_2_mix_aiv
  - 重新计算生成目标文件的 SHA256 并回填产物 JSON 描述符（CANN 依此校验 .o 完整性）
  - 同步安装 kernel/config、sageattention2.json、binary_info_config.json 与 aic-ops-info.json
  - 统一置备 vendors/customize 与 vendors/custom_transformer 双 vendor 目录并配置 config.ini

源模板 kernel_meta/kernel/<soc>/sageattention2/*.json 是手写的 dtype 分派表，
binary_info_config.json 按文件名硬引用它们，不可由本脚本生成，需随源码入库。
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct


# ELF 字段宽度随 32/64 位变化，统一封装避免散落的条件格式串
_SHDR_FMT = {True: "<IIQQQQIIQQ", False: "<IIIIII"}
_SHDR_SIZE = {True: 64, False: 24}
_SHNUM_OFF = {True: (58, 60, 62), False: (46, 48, 50)}  # e_shentsize/e_shnum/e_shstrndx
AIMACHINE_ASCEND = 0x1029
_NO_OFFSET = 1 << 62           # 未布局 section 的排序哨兵


def _read_addr(data, is_64, off64, off32):
    """读取一个 8 字节(64 位) / 4 字节(32 位) 的偏移或地址字段。"""
    return struct.unpack("<Q" if is_64 else "<I",
                         data[off64:off64 + 8] if is_64 else data[off32:off32 + 4])[0]


def _read_half(data, is_64, idx):
    """读取第 idx 个 2 字节字段(e_shentsize / e_shnum / e_shstrndx)。"""
    off = _SHNUM_OFF[is_64][idx]
    return struct.unpack("<H", data[off:off + 2])[0]


def _unpack_shdr(data, off, is_64):
    """解析一个 Section Header，返回 10 元组。"""
    return struct.unpack(_SHDR_FMT[is_64], data[off:off + _SHDR_SIZE[is_64]])


def extract_aicore_binary(data):
    """从输入 ELF 数据中提取纯设备 ELF (e_machine == 0x1029)"""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError("Input data is not a valid ELF file")
    is_64 = (data[4] == 2)
    _, e_machine = struct.unpack("<HH", data[16:20])
    if e_machine == AIMACHINE_ASCEND:
        # 已经是 Ascend AI Core 设备 ELF
        return bytes(data)

    # 宿主包含异构目标，在 section 表中查找 .aicore_binary
    e_shoff = _read_addr(data, is_64, 40, 32)
    e_shentsize = _read_half(data, is_64, 0)
    e_shnum = _read_half(data, is_64, 1)
    e_shstrndx = _read_half(data, is_64, 2)

    # 读取 shstrtab
    shstr_entry = e_shoff + e_shstrndx * e_shentsize
    shdr_str = _unpack_shdr(data, shstr_entry, is_64)
    shstr_offset, shstr_size = shdr_str[4], shdr_str[5]
    shstrtab = data[shstr_offset:shstr_offset + shstr_size]

    def get_sec_name(name_idx):
        end = shstrtab.find(b"\0", name_idx)
        return shstrtab[name_idx:end].decode("latin1", errors="ignore")

    for i in range(e_shnum):
        shdr = _unpack_shdr(data, e_shoff + i * e_shentsize, is_64)
        sh_name, sec_offset, sec_size = shdr[0], shdr[4], shdr[5]
        if get_sec_name(sh_name) == ".aicore_binary":
            return bytes(data[sec_offset:sec_offset + sec_size])

    raise ValueError("Could not find .aicore_binary section in host ELF")


def generate_dual_target_elf(base_dev_elf_bytes, bin_name):
    """
    为指定 target bin_name 生成同时包含 tilingKey: 1 (_1, FP16) 与 tilingKey: 2 (_2, BF16)
    的混核符号与元数据 section 的 Ascend C 设备 ELF。
    单一内核通过动态 TilingData 中的 dtype_mode (NormDtypeMode) 自动分支处理 FP16 与 BF16。
    """
    dev_elf = bytearray(base_dev_elf_bytes)
    assert dev_elf[:4] == b"\x7fELF" and dev_elf[4] == 2, "Only 64-bit ELF supported"

    k1 = f"{bin_name}_1"
    k2 = f"{bin_name}_2"

    e_shoff = struct.unpack("<Q", dev_elf[40:48])[0]
    e_shentsize = struct.unpack("<H", dev_elf[58:60])[0]
    e_shnum = struct.unpack("<H", dev_elf[60:62])[0]
    e_shstrndx = struct.unpack("<H", dev_elf[62:64])[0]

    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name, sh_type, sh_flags, sh_addr, sec_offset, sec_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack("<IIQQQQIIQQ", dev_elf[off:off+64])
        shdrs.append({
            "idx": i, "name_idx": sh_name, "type": sh_type, "flags": sh_flags,
            "addr": sh_addr, "offset": sec_offset, "size": sec_size,
            "link": sh_link, "info": sh_info, "addralign": sh_addralign, "entsize": sh_entsize,
            "data": bytearray(dev_elf[sec_offset:sec_offset + sec_size])
        })

    shstr_sec = shdrs[e_shstrndx]
    def get_shname(idx):
        end = shstr_sec["data"].find(b"\0", idx)
        return shstr_sec["data"][idx:end].decode("latin1")

    for s in shdrs:
        s["name"] = get_shname(s["name_idx"])

    # 查找原始 AIC 与 AIV 元数据 section
    aic_meta_sec = next(s for s in shdrs if s["name"] == ".ascend.meta.sage_attention2_mix_aic")
    aiv_meta_sec = next(s for s in shdrs if s["name"] == ".ascend.meta.sage_attention2_mix_aiv")

    def add_shstr(name_str):
        offset = len(shstr_sec["data"])
        shstr_sec["data"].extend(name_str.encode("latin1") + b"\0")
        return offset

    # 1. 将 section 原位重命名为 _1 (FP16)
    aic_meta_sec["name"] = f".ascend.meta.{k1}_mix_aic"
    aic_meta_sec["name_idx"] = add_shstr(aic_meta_sec["name"])

    aiv_meta_sec["name"] = f".ascend.meta.{k1}_mix_aiv"
    aiv_meta_sec["name_idx"] = add_shstr(aiv_meta_sec["name"])

    # 2. 克隆 section 为 _2 (BF16)
    # 2. 克隆 section 为 _2 (BF16)，并登记新旧 section 索引供符号表重定位使用
    k2_meta_secs = []
    for src_sec, kind in ((aic_meta_sec, "aic"), (aiv_meta_sec, "aiv")):
        name = f".ascend.meta.{k2}_mix_{kind}"
        clone = dict(src_sec, name=name, name_idx=add_shstr(name),
                     idx=len(shdrs), offset=0, data=bytearray(src_sec["data"]))
        shdrs.append(clone)
        k2_meta_secs.append(clone)
    k2_aic_meta_sec, k2_aiv_meta_sec = k2_meta_secs

    shstr_sec["size"] = len(shstr_sec["data"])

    # 3. 处理 symtab 与 strtab
    symtab_sec = next(s for s in shdrs if s["type"] == 2)
    strtab_sec = shdrs[symtab_sec["link"]]

    def add_str(str_val):
        offset = len(strtab_sec["data"])
        strtab_sec["data"].extend(str_val.encode("latin1") + b"\0")
        return offset

    entsize = symtab_sec["entsize"] or 24
    orig_sym_count = len(symtab_sec["data"]) // entsize
    orig_sh_info = symtab_sec["info"]

    local_syms = []
    global_syms = []
    new_local_syms = []
    new_global_syms = []

    for si in range(orig_sym_count):
        raw = symtab_sec["data"][si*entsize:(si+1)*entsize]
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack("<IBBHQQ", raw)
        end = strtab_sec["data"].find(b"\0", st_name)
        sym_name = strtab_sec["data"][st_name:end].decode("latin1")

        is_local = (si < orig_sh_info)

        if "sage_attention2" in sym_name:
            k1_name = sym_name.replace("sage_attention2", k1)
            k2_name = sym_name.replace("sage_attention2", k2)
            st_name = add_str(k1_name)
            k2_st_name = add_str(k2_name)

            k2_shndx = st_shndx
            if st_shndx == aic_meta_sec["idx"]:
                k2_shndx = k2_aic_meta_sec["idx"]
            elif st_shndx == aiv_meta_sec["idx"]:
                k2_shndx = k2_aiv_meta_sec["idx"]

            k2_entry = struct.pack("<IBBHQQ", k2_st_name, st_info, st_other, k2_shndx, st_value, st_size)
            if is_local:
                new_local_syms.append(k2_entry)
            else:
                new_global_syms.append(k2_entry)

        entry = struct.pack("<IBBHQQ", st_name, st_info, st_other, st_shndx, st_value, st_size)
        if is_local:
            local_syms.append(entry)
        else:
            global_syms.append(entry)

    # 为 2 个新克隆的 metadata section 添加 STT_SECTION 本地符号
    stt_sec_k2_aic = struct.pack("<IBBHQQ", 0, 3, 0, k2_aic_meta_sec["idx"], 0, 0)
    stt_sec_k2_aiv = struct.pack("<IBBHQQ", 0, 3, 0, k2_aiv_meta_sec["idx"], 0, 0)
    new_local_syms.append(stt_sec_k2_aic)
    new_local_syms.append(stt_sec_k2_aiv)

    shift = len(new_local_syms)

    # 4. 同步修正 SHT_RELA 重定位表项中的符号引用索引
    for s in shdrs:
        if s["type"] == 4:  # SHT_RELA
            rela_entsize = s["entsize"] or 24
            rela_count = len(s["data"]) // rela_entsize
            new_relas = []
            for ri in range(rela_count):
                r_offset, r_info, r_addend = struct.unpack("<QQq", s["data"][ri*rela_entsize:(ri+1)*rela_entsize])
                sym_idx = r_info >> 32
                r_type = r_info & 0xffffffff
                if sym_idx >= orig_sh_info:
                    sym_idx += shift
                new_relas.append(struct.pack("<QQq", r_offset, (sym_idx << 32) | r_type, r_addend))
            s["data"] = bytearray(b"".join(new_relas))
            s["size"] = len(s["data"])

    # 5. 组合新符号表（确保所有 LOCAL 符号在 GLOBAL/WEAK 符号前，info 为首个非 LOCAL 符号的索引）
    all_syms = local_syms + new_local_syms + global_syms + new_global_syms
    symtab_sec["data"] = bytearray(b"".join(all_syms))
    symtab_sec["size"] = len(symtab_sec["data"])
    symtab_sec["info"] = len(local_syms) + len(new_local_syms)
    strtab_sec["size"] = len(strtab_sec["data"])

    # 6. 重组 ELF 布局并回写 Section Header Table
    out = bytearray(dev_elf[:64])
    e_phoff = struct.unpack("<Q", dev_elf[32:40])[0]
    e_phnum = struct.unpack("<H", dev_elf[56:58])[0]
    e_phentsize = struct.unpack("<H", dev_elf[54:56])[0]
    if e_phnum > 0 and e_phoff > 0:
        out.extend(dev_elf[64:e_phoff + e_phnum * e_phentsize])

    # 保留原有 section 顺序，未布局(offset==0)的排到末尾
    sorted_shdrs = sorted(shdrs, key=lambda s: s["offset"] or _NO_OFFSET)
    for s in sorted_shdrs:
        if s["type"] == 0:
            s["offset"] = 0
            continue
        align = max(s["addralign"], 1)
        pad = (align - (len(out) % align)) % align
        out.extend(b"\0" * pad)
        s["offset"] = len(out)
        out.extend(s["data"])

    pad = (8 - (len(out) % 8)) % 8
    out.extend(b"\0" * pad)
    new_e_shoff = len(out)

    for s in shdrs:
        hdr = struct.pack("<IIQQQQIIQQ", s["name_idx"], s["type"], s["flags"], s["addr"], s["offset"], s["size"], s["link"], s["info"], s["addralign"], s["entsize"])
        out.extend(hdr)

    out[40:48] = struct.pack("<Q", new_e_shoff)
    out[60:62] = struct.pack("<H", len(shdrs))

    return bytes(out)


def generate_single_target(base_dev_elf_bytes, bin_name, out_o_path):
    """为指定 target 生成同时包含 _1 (FP16) 与 _2 (BF16) 的双符号与元数据 section 的 .o 文件"""
    transformed = generate_dual_target_elf(base_dev_elf_bytes, bin_name)
    with open(out_o_path, "wb") as f:
        f.write(transformed)


def vendor_dispatch_dir(vendors_root, vendor, soc):
    """返回某vendor 内 static target(.o/.json) 的分发目录。"""
    return os.path.join(vendors_root, vendor, "op_impl", "ai_core", "tbe",
                        "kernel", soc, "sageattention2")


def dispatch_kernel_binaries(kernel_obj_path, meta_dir, output_vendors_dir, soc="ascend950",
                             sync_meta_sha256=False):
    """
    核心调度分发主流程。

    sync_meta_sha256: 是否把生成 .o 的新 sha256 回写到 meta-dir 源模板。
      默认 False —— 源模板是手写的 dtype 分派表（binary_info_config.json 按文件名硬引用它们），
      其中的 sha256 在每次构建时都会被无条件重算覆盖，从不被读取校验，回写只会让源码树
      每次编译都产生 16 个文件的无关 diff。仅在需要固化产物校验和时显式开启。
    """
    if not os.path.isfile(kernel_obj_path):
        raise FileNotFoundError(f"Compiled kernel object not found: {kernel_obj_path}")

    with open(kernel_obj_path, "rb") as f:
        raw_kernel = f.read()

    base_dev_elf = extract_aicore_binary(raw_kernel)
    print(f"[dispatch] Extracted base device ELF: {len(base_dev_elf)} bytes (magic=0x1029)")

    template_kernel_dir = os.path.join(meta_dir, "kernel", soc, "sageattention2")
    json_templates = sorted(glob.glob(os.path.join(template_kernel_dir, "*.json")))
    if not json_templates:
        raise FileNotFoundError(f"No json templates found in {template_kernel_dir}")

    print(f"[dispatch] Found {len(json_templates)} static dispatch JSON descriptors")

    # 静态配置与 ops-info：源模板 -> vendor 内目标目录（缺失即跳过，非致命）
    static_configs = [
        (os.path.join("kernel", "config", soc, "binary_info_config.json"), "kernel_config"),
        (os.path.join("kernel", "config", soc, "sageattention2.json"), "kernel_config"),
        (os.path.join("config", soc, f"aic-{soc}-ops-info.json"), "op_config"),
        (os.path.join("config", soc, "sageattention2.ini"), "op_config"),
    ]

    # 双 Vendor 目录统一置备：customize 与 custom_transformer
    vendor_names = ["customize", "custom_transformer"]
    for vname in vendor_names:
        vdir = os.path.join(output_vendors_dir, vname)
        k_soc_dir = os.path.join(vdir, "op_impl", "ai_core", "tbe", "kernel", soc)
        k_disp_dir = vendor_dispatch_dir(output_vendors_dir, vname, soc)
        cfg_dirs = {
            "kernel_config": os.path.join(vdir, "op_impl", "ai_core", "tbe", "kernel", "config", soc),
            "op_config": os.path.join(vdir, "op_impl", "ai_core", "tbe", "config", soc),
        }

        for d in (k_disp_dir, *cfg_dirs.values()):
            os.makedirs(d, exist_ok=True)

        # 写入根级 sageattention2.o
        with open(os.path.join(k_soc_dir, "sageattention2.o"), "wb") as f:
            f.write(base_dev_elf)

        # 同步静态配置与 ops-info
        for rel_path, cfg_key in static_configs:
            src = os.path.join(meta_dir, rel_path)
            if os.path.isfile(src):
                shutil.copy2(src, cfg_dirs[cfg_key])

        # 生成每个 static target 的 .o 与配套 .json
        for jf in json_templates:
            with open(jf, "r") as fp:
                jdata = json.load(fp)

            bin_name = jdata["binFileName"]

            # 标准 CANN 规范：严格包含 tilingKey: 1 (_1, FP16) 与 tilingKey: 2 (_2, BF16)
            jdata["kernelList"] = [
                {
                    "tilingKey": tiling_key,
                    "kernelType": "MIX_AIC",
                    "crossCoreSync": 1,
                    "taskRation": "1:2",
                    "kernelName": f"{bin_name}_{tiling_key}",
                }
                for tiling_key in (1, 2)
            ]
            jdata["taskRation"] = "tilingKey"

            out_o = os.path.join(k_disp_dir, f"{bin_name}.o")
            out_json = os.path.join(k_disp_dir, f"{bin_name}.json")

            # 首个 vendor 生成 .o，其余 vendor 直接复制（同内容，避免重复 ELF 重写）
            if vname == vendor_names[0]:
                generate_single_target(base_dev_elf, bin_name, out_o)
            else:
                shutil.copy2(os.path.join(vendor_dispatch_dir(output_vendors_dir, vendor_names[0], soc),
                                        f"{bin_name}.o"), out_o)

            with open(out_o, "rb") as fp:
                # CANN 依此校验 .o 完整性，必须与实际产物一致
                jdata["sha256"] = hashlib.sha256(fp.read()).hexdigest()

            with open(out_json, "w") as fp:
                json.dump(jdata, fp, indent=1)
                fp.write("\n")

            # 可选：把新sha256 固化回源模板（默认关闭，见 sync_meta_sha256）
            if sync_meta_sha256 and vname == vendor_names[0]:
                with open(jf, "w") as fp:
                    json.dump(jdata, fp, indent=1)
                    fp.write("\n")

        print(f"[dispatch] Successfully dispatched {len(json_templates)} binaries to {vname}: {k_disp_dir}")

    # 写入 vendors/config.ini 配置双优先级
    config_ini_path = os.path.join(output_vendors_dir, "config.ini")
    with open(config_ini_path, "w") as f:
        f.write("[common]\nload_priority=customize,custom_transformer\n")
    print(f"[dispatch] Generated {config_ini_path} (load_priority=customize,custom_transformer)")


def main():
    parser = argparse.ArgumentParser(description="SageAttention2 CANN ACLNN Static Binary Dispatcher")
    parser.add_argument("--kernel-obj", required=True, help="Path to compiled kernel object (sageattention2_bisheng.o)")
    parser.add_argument("--meta-dir", required=True, help="Path to kernel_meta directory containing JSON templates")
    parser.add_argument("--output-dir", required=True, help="Path to output vendors root directory (e.g. build/opp/vendors)")
    parser.add_argument("--soc", default="ascend950", help="Target SoC version (default: ascend950)")
    parser.add_argument("--sync-meta-sha256", action="store_true",
                        help="Also write the freshly computed sha256 back into the meta-dir source "
                             "templates. Off by default: the templates are hand-written dispatch "
                             "descriptors and their sha256 is always recomputed, so syncing only "
                             "creates unrelated diffs in the source tree.")
    args = parser.parse_args()

    dispatch_kernel_binaries(
        kernel_obj_path=args.kernel_obj,
        meta_dir=args.meta_dir,
        output_vendors_dir=args.output_dir,
        soc=args.soc,
        sync_meta_sha256=args.sync_meta_sha256
    )


if __name__ == "__main__":
    main()
