"""Replace a path prefix in an ASC-style batch case list."""


def replace_path(paths, old_prefix, new_prefix):
    return [path.replace(old_prefix, new_prefix, 1) for path in paths]

