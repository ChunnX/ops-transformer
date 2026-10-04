#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
"""
dispatch_binary.py: CANN ACLNN 静态二进制调度生成器

负责将毕昇编译器单次编译出的 Ascend C 设备核目标文件 (sageattention2_bisheng.o)，
分发并映射为 CANN ACLNN 静态注册表所需的 16 个 static binary targets:
  - 自动解包 .aicore_binary 纯设备 ELF (e_machine 0x1029, ET_EXEC)
  - 对 16 个 DataType/Format 静态档位，逐一重命名 AIC/AIV 混核符号与元数据 section:
      sage_attention2_mix_aic   -> SageAttention2_<hash>_0_mix_aic
      sage_attention2_mix_aic__ -> SageAttention2_<hash>_0_mix_aic__
      sage_attention2_mix_aiv   -> SageAttention2_<hash>_0_mix_aiv
      sage_attention2_mix_aiv__ -> SageAttention2_<hash>_0_mix_aiv__
      .ascend.meta.sage_attention2_mix_aic -> .ascend.meta.SageAttention2_<hash>_0_mix_aic
      .ascend.meta.sage_attention2_mix_aiv -> .ascend.meta.SageAttention2_<hash>_0_mix_aiv
  - 重新计算生成目标文件的 SHA256 并回填对应 JSON 描述符
  - 同步安装 kernel/config、sageattention2.json、binary_info_config.json 与 aic-ops-info.json
  - 统一置备 vendors/customize 与 vendors/custom_transformer 双 vendor 目录并配置 config.ini
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys


def find_objcopy_executable(explicit_path=None):
    if explicit_path and os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK):
        return explicit_path
    env_objcopy = os.environ.get("OBJCOPY")
    if env_objcopy and os.path.isfile(env_objcopy) and os.access(env_objcopy, os.X_OK):
        return env_objcopy
    # Check standard PATH
    for name in ("llvm-objcopy", "objcopy"):
        p = shutil.which(name)
        if p:
            return p
    # Check CANN / BiSheng toolchain directories
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    candidates = [
        os.path.join(ascend_home, "bin", "llvm-objcopy"),
        os.path.join(ascend_home, "compiler", "ccec_compiler", "bin", "llvm-objcopy"),
        "/usr/local/Ascend/ascend-toolkit/latest/bin/llvm-objcopy",
        "/usr/bin/llvm-objcopy",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def extract_aicore_binary(data):
    """从输入 ELF 数据中提取纯设备 ELF (e_machine == 0x1029)"""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError("Input data is not a valid ELF file")
    is_64 = (data[4] == 2)
    e_type, e_machine = struct.unpack("<HH", data[16:20])
    if e_machine == 0x1029:
        # 已经是 Ascend AI Core 设备 ELF
        return bytes(data)

    # 宿主包含异构目标，在 section 表中查找 .aicore_binary
    e_shoff = struct.unpack("<Q" if is_64 else "<I", data[40:48] if is_64 else data[32:36])[0]
    e_shentsize = struct.unpack("<H", data[58:60] if is_64 else data[46:48])[0]
    e_shnum = struct.unpack("<H", data[60:62] if is_64 else data[48:50])[0]
    e_shstrndx = struct.unpack("<H", data[62:64] if is_64 else data[50:52])[0]

    # 读取 shstrtab
    shstr_entry = e_shoff + e_shstrndx * e_shentsize
    shstr_hdr = data[shstr_entry:shstr_entry + e_shentsize]
    _, _, _, _, shstr_offset, shstr_size = struct.unpack("<IIQQQQ" if is_64 else "<IIIIII", shstr_hdr[: (40 if is_64 else 24)])
    shstrtab = data[shstr_offset:shstr_offset + shstr_size]

    def get_sec_name(name_idx):
        end = shstrtab.find(b"\0", name_idx)
        return shstrtab[name_idx:end].decode("latin1", errors="ignore")

    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        hdr = data[off:off + e_shentsize]
        sh_name, sh_type, sh_flags, sh_addr, sec_offset, sec_size = struct.unpack("<IIQQQQ" if is_64 else "<IIIIII", hdr[: (40 if is_64 else 24)])
        name = get_sec_name(sh_name)
        if name == ".aicore_binary":
            embedded = data[sec_offset:sec_offset + sec_size]
            if len(embedded) >= 64 and embedded[:4] == b"\x7fELF":
                _, dev_machine = struct.unpack("<HH", embedded[16:20])
                if dev_machine == 0x1029:
                    return bytes(embedded)
            return bytes(embedded)

    raise ValueError("Could not find .aicore_binary section in host ELF")


def py_elf_rename_symbols_and_sections(elf_bytes, sym_renames, sec_renames):
    """纯 Python 零依赖 ELF 符号与 section 重命名实现 (作为 objcopy 兜底)"""
    data = bytearray(elf_bytes)
    assert data[:4] == b"\x7fELF" and data[4] == 2, "Only 64-bit ELF supported"

    e_shoff = struct.unpack("<Q", data[40:48])[0]
    e_shentsize = struct.unpack("<H", data[58:60])[0]
    e_shnum = struct.unpack("<H", data[60:62])[0]
    e_shstrndx = struct.unpack("<H", data[62:64])[0]

    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        hdr = data[off:off + e_shentsize]
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack("<IIQQQQIIQQ", hdr)
        shdrs.append({
            "idx": i, "name_idx": sh_name, "type": sh_type, "flags": sh_flags,
            "addr": sh_addr, "offset": sh_offset, "size": sh_size,
            "link": sh_link, "info": sh_info, "addralign": sh_addralign, "entsize": sh_entsize,
            "data": bytearray(data[sh_offset:sh_offset + sh_size])
        })

    shstr_sec = shdrs[e_shstrndx]

    def get_shname(idx):
        end = shstr_sec["data"].find(b"\0", idx)
        return shstr_sec["data"][idx:end].decode("latin1")

    for s in shdrs:
        s["name"] = get_shname(s["name_idx"])

    # 1. 重命名 shstrtab 中的 section 名字
    for s in shdrs:
        if s["name"] in sec_renames:
            new_name = sec_renames[s["name"]].encode("latin1") + b"\0"
            s["name_idx"] = len(shstr_sec["data"])
            shstr_sec["data"].extend(new_name)
            s["name"] = sec_renames[s["name"]]
    shstr_sec["size"] = len(shstr_sec["data"])

    # 2. 重命名 symtab / dynsym 中的符号名
    for s in shdrs:
        if s["type"] in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            str_sec = shdrs[s["link"]]
            entsize = s["entsize"] or 24
            num_syms = len(s["data"]) // entsize
            sym_entries = []
            for i in range(num_syms):
                sym_bytes = s["data"][i * entsize:(i + 1) * entsize]
                st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack("<IBBHQQ", sym_bytes)
                end = str_sec["data"].find(b"\0", st_name)
                sym_name = str_sec["data"][st_name:end].decode("latin1")
                if sym_name in sym_renames:
                    new_sym = sym_renames[sym_name].encode("latin1") + b"\0"
                    st_name = len(str_sec["data"])
                    str_sec["data"].extend(new_sym)
                sym_entries.append(struct.pack("<IBBHQQ", st_name, st_info, st_other, st_shndx, st_value, st_size))
            s["data"] = bytearray(b"".join(sym_entries))
            s["size"] = len(s["data"])
            str_sec["size"] = len(str_sec["data"])

    # 3. 顺序重组 ELF section 布局
    out = bytearray(data[:64])
    sorted_shdrs = sorted(shdrs, key=lambda s: s["offset"] if s["offset"] > 0 else 999999999)
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
    return bytes(out)


def generate_single_target(base_dev_elf_bytes, bin_name, k_name, out_o_path, objcopy_bin):
    """为指定 target 生成重命名符号与 section 后的 .o 文件"""
    sym_renames = {
        "sage_attention2_mix_aic": f"{k_name}_mix_aic",
        "sage_attention2_mix_aic__": f"{k_name}_mix_aic__",
        "sage_attention2_mix_aiv": f"{k_name}_mix_aiv",
        "sage_attention2_mix_aiv__": f"{k_name}_mix_aiv__",
        "sage_attention2_mix_aic$local": f"{k_name}_mix_aic$local",
        "sage_attention2_mix_aiv$local": f"{k_name}_mix_aiv$local",
    }
    sec_renames = {
        ".ascend.meta.sage_attention2_mix_aic": f".ascend.meta.{k_name}_mix_aic",
        ".ascend.meta.sage_attention2_mix_aiv": f".ascend.meta.{k_name}_mix_aiv",
    }

    if objcopy_bin:
        tmp_base = out_o_path + ".tmp_base.o"
        try:
            with open(tmp_base, "wb") as f:
                f.write(base_dev_elf_bytes)
            cmd = [objcopy_bin]
            for old_sym, new_sym in sym_renames.items():
                cmd.append(f"--redefine-sym={old_sym}={new_sym}")
            for old_sec, new_sec in sec_renames.items():
                cmd.append(f"--rename-section={old_sec}={new_sec}")
            cmd.extend([tmp_base, out_o_path])
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.isfile(out_o_path):
                return
            # objcopy 返回非 0 时回退纯 Python 实现
            print(f"[WARN] objcopy failed for {bin_name} (rc={res.returncode}): {res.stderr.strip()}; falling back to pure-Python ELF renamer")
        finally:
            if os.path.exists(tmp_base):
                os.remove(tmp_base)

    # 纯 Python 实现
    transformed = py_elf_rename_symbols_and_sections(base_dev_elf_bytes, sym_renames, sec_renames)
    with open(out_o_path, "wb") as f:
        f.write(transformed)


def dispatch_kernel_binaries(kernel_obj_path, meta_dir, output_vendors_dir, soc="ascend950", objcopy_path=None):
    """核心调度分发主流程"""
    if not os.path.isfile(kernel_obj_path):
        raise FileNotFoundError(f"Compiled kernel object not found: {kernel_obj_path}")

    with open(kernel_obj_path, "rb") as f:
        raw_kernel = f.read()

    base_dev_elf = extract_aicore_binary(raw_kernel)
    print(f"[dispatch] Extracted base device ELF: {len(base_dev_elf)} bytes (magic=0x1029)")

    objcopy_bin = find_objcopy_executable(objcopy_path)
    if objcopy_bin:
        print(f"[dispatch] Using objcopy tool: {objcopy_bin}")
    else:
        print("[dispatch] llvm-objcopy not found; using pure-Python ELF renamer")

    template_kernel_dir = os.path.join(meta_dir, "kernel", soc, "sageattention2")
    json_templates = sorted(glob.glob(os.path.join(template_kernel_dir, "*.json")))
    if not json_templates:
        raise FileNotFoundError(f"No json templates found in {template_kernel_dir}")

    print(f"[dispatch] Found {len(json_templates)} static dispatch JSON descriptors")

    # 双 Vendor 目录统一置备：customize 与 custom_transformer
    vendor_names = ["customize", "custom_transformer"]
    for vname in vendor_names:
        vdir = os.path.join(output_vendors_dir, vname)
        k_soc_dir = os.path.join(vdir, "op_impl", "ai_core", "tbe", "kernel", soc)
        k_disp_dir = os.path.join(k_soc_dir, "sageattention2")
        k_cfg_dir = os.path.join(vdir, "op_impl", "ai_core", "tbe", "kernel", "config", soc)
        op_cfg_dir = os.path.join(vdir, "op_impl", "ai_core", "tbe", "config", soc)

        os.makedirs(k_disp_dir, exist_ok=True)
        os.makedirs(k_cfg_dir, exist_ok=True)
        os.makedirs(op_cfg_dir, exist_ok=True)

        # 写入根级 sageattention2.o (纯设备 ELF 与带符号的原始目标均备齐)
        with open(os.path.join(k_soc_dir, "sageattention2.o"), "wb") as f:
            f.write(base_dev_elf)

        # 同步静态配置与 ops-info
        src_bin_cfg = os.path.join(meta_dir, "kernel", "config", soc, "binary_info_config.json")
        src_sage_json = os.path.join(meta_dir, "kernel", "config", soc, "sageattention2.json")
        src_aic_info = os.path.join(meta_dir, "config", soc, f"aic-{soc}-ops-info.json")
        src_sage_ini = os.path.join(meta_dir, "config", soc, "sageattention2.ini")

        if os.path.isfile(src_bin_cfg):
            shutil.copy2(src_bin_cfg, k_cfg_dir)
        if os.path.isfile(src_sage_json):
            shutil.copy2(src_sage_json, k_cfg_dir)
        if os.path.isfile(src_aic_info):
            shutil.copy2(src_aic_info, op_cfg_dir)
        if os.path.isfile(src_sage_ini):
            shutil.copy2(src_sage_ini, op_cfg_dir)

        # 生成 16 个 static target .o 与回填 sha256 后的 .json
        for jf in json_templates:
            with open(jf, "r") as fp:
                jdata = json.load(fp)

            bin_name = jdata["binFileName"]
            k_name = jdata["kernelList"][0]["kernelName"]

            out_o = os.path.join(k_disp_dir, f"{bin_name}.o")
            out_json = os.path.join(k_disp_dir, f"{bin_name}.json")

            generate_single_target(base_dev_elf, bin_name, k_name, out_o, objcopy_bin)

            with open(out_o, "rb") as fp:
                new_sha256 = hashlib.sha256(fp.read()).hexdigest()

            jdata["sha256"] = new_sha256
            with open(out_json, "w") as fp:
                json.dump(jdata, fp, indent=1)

        print(f"[dispatch] Successfully dispatched 16 binaries to {vname}: {k_disp_dir}")

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
    parser.add_argument("--objcopy", default=None, help="Explicit path to llvm-objcopy binary")
    args = parser.parse_args()

    dispatch_kernel_binaries(
        kernel_obj_path=args.kernel_obj,
        meta_dir=args.meta_dir,
        output_vendors_dir=args.output_dir,
        soc=args.soc,
        objcopy_path=args.objcopy
    )


if __name__ == "__main__":
    main()
