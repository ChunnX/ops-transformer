#!/usr/bin/env python3
"""
Rename sparse_lightning_indexer_kl_loss_grad files to dense_lightning_indexer_kl_loss_grad
and update all internal references.
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Define all files that need renaming
FILES_TO_RENAME = []
for root, dirs, files in os.walk(BASE_DIR):
    # Skip hidden directories and __pycache__
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for f in files:
        if f == 'rename_sparse_to_dense.py':
            continue
        if 'sparse_lightning_indexer_kl_loss_grad' in f.lower() or \
           'sparselightningindexerkllossgrad' in f.lower():
            old_path = os.path.join(root, f)
            # Generate new filename
            new_name = f.replace('sparse_lightning_indexer_kl_loss_grad', 'dense_lightning_indexer_kl_loss_grad')
            new_name = new_name.replace('SparseLightningIndexerKLLossGrad', 'DenseLightningIndexerKLLossGrad')
            new_name = new_name.replace('sparse_lightning_indexer_kl_loss_grad', 'dense_lightning_indexer_kl_loss_grad')
            new_path = os.path.join(root, new_name)
            if old_path != new_path:
                FILES_TO_RENAME.append((old_path, new_path))

print(f"Found {len(FILES_TO_RENAME)} files to rename:")
for old, new in sorted(FILES_TO_RENAME):
    rel_old = os.path.relpath(old, BASE_DIR)
    rel_new = os.path.relpath(new, BASE_DIR)
    print(f"  {rel_old} -> {rel_new}")

# Step 2: Content replacements (order matters - longer strings first)
CONTENT_REPLACEMENTS = [
    # Include guards (uppercase)
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_GENERAL_REGBASE_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_GENERAL_REGBASE_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_COMMON_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_COMMON_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_GENERAL_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_GENERAL_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TEMPLATE_TILING_KEY_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TEMPLATE_TILING_KEY_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_BASE_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_COMMON_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_COMMON_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR2_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR2_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_SERVICE_CUBE_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_SERVICE_CUBE_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_KERNEL_BASE_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_KERNEL_BASE_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_REGBASE_COMMON_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_REGBASE_COMMON_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_REGBASE_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TILING_REGBASE_H'),
    ('SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TEMPLATE_TILING_KEY_REGBASE_H',
     'DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_TEMPLATE_TILING_KEY_REGBASE_H'),

    # Class/struct/function names (PascalCase)
    ('SparseLightningIndexerKLLossGrad', 'DenseLightningIndexerKLLossGrad'),
    ('SparseLightningIndexerGradRegBaseTilingData', 'DenseLightningIndexerGradRegBaseTilingData'),
    ('SparseLightningIndexerKLLossGradCompileInfo', 'DenseLightningIndexerKLLossGradCompileInfo'),
    ('SparseLightningIndexerKLLossGradTilingBase', 'DenseLightningIndexerKLLossGradTilingBase'),
    ('SparseLightningIndexerKLLossGradTilingData', 'DenseLightningIndexerKLLossGradTilingData'),
    ('SparseLightningIndexerKLLossGradTilingGeneralRegbase', 'DenseLightningIndexerKLLossGradTilingGeneralRegbase'),
    ('SparseLightningIndexerGradCompileInfo', 'DenseLightningIndexerGradCompileInfo'),
    ('InferShapeSparseLightningIndexerKLLossGrad', 'InferShapeDenseLightningIndexerKLLossGrad'),
    ('InferDataTypeSparseLightningIndexerKLLossGrad', 'InferDataTypeDenseLightningIndexerKLLossGrad'),
    ('TilingSparseLightningIndexerKLLossGrad', 'TilingDenseLightningIndexerKLLossGrad'),
    ('TilingPrepareForSparseLightningIndexerKLLossGrad', 'TilingPrepareForDenseLightningIndexerKLLossGrad'),

    # File references in #include and \file comments (lowercase)
    ('sparse_lightning_indexer_kl_loss_grad', 'dense_lightning_indexer_kl_loss_grad'),

    # CMake variable names
    ('sparse_lightning_indexer_kl_loss_grad_depends', 'dense_lightning_indexer_kl_loss_grad_depends'),
]


def process_file_content(content):
    """Apply all content replacements."""
    for old, new in CONTENT_REPLACEMENTS:
        content = content.replace(old, new)
    return content


# Step 3: Process all files
def main():
    processed_count = 0
    renamed_count = 0

    # First, collect all files that exist (before any renaming)
    all_files = set()
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f == 'rename_sparse_to_dense.py':
                continue
            all_files.add(os.path.join(root, f))

    # Process content of all files first (before renaming)
    file_contents = {}
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            new_content = process_file_content(content)
            if new_content != content:
                file_contents[filepath] = new_content
        except Exception as e:
            print(f"  Warning: Could not read {filepath}: {e}")

    print(f"\nFiles with content changes: {len(file_contents)}")

    # Write updated content back to original files first
    for filepath, new_content in file_contents.items():
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            rel = os.path.relpath(filepath, BASE_DIR)
            print(f"  Updated content: {rel}")
            processed_count += 1
        except Exception as e:
            print(f"  Warning: Could not write {filepath}: {e}")

    # Now rename files
    print(f"\nRenaming files:")
    rename_map = {old: new for old, new in FILES_TO_RENAME}

    # Rename in reverse order to avoid conflicts (deepest paths first)
    for old_path, new_path in sorted(FILES_TO_RENAME, key=lambda x: x[0], reverse=True):
        if os.path.exists(old_path):
            if old_path != new_path:
                try:
                    # If target already exists (from content update), remove it
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(old_path, new_path)
                    rel_old = os.path.relpath(old_path, BASE_DIR)
                    rel_new = os.path.relpath(new_path, BASE_DIR)
                    print(f"  Renamed: {rel_old} -> {rel_new}")
                    renamed_count += 1
                except Exception as e:
                    print(f"  Warning: Could not rename {old_path} -> {new_path}: {e}")
                    # Fallback: copy and delete
                    try:
                        shutil.copy2(old_path, new_path)
                        os.remove(old_path)
                        rel_old = os.path.relpath(old_path, BASE_DIR)
                        rel_new = os.path.relpath(new_path, BASE_DIR)
                        print(f"  Copied+deleted: {rel_old} -> {rel_new}")
                        renamed_count += 1
                    except Exception as e2:
                        print(f"  Error: Could not copy+delete: {e2}")

    print(f"\n=== Summary ===")
    print(f"Content updated in {processed_count} files")
    print(f"Renamed {renamed_count} files")
    print(f"Done!")


if __name__ == '__main__':
    main()