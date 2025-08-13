#!/usr/bin/env python3

"""
provides functionality related to saving data and artifacts
"""

import io
import gzip
import functools
import os
import pathlib
import typing
import warnings


def set_perms(saver: typing.Callable) -> typing.Callable:
    """sets 770 permissions and group ownership to that of the enclosing folder during save"""

    @functools.wraps(saver)
    def wrapper(file, *args, **kwargs):
        f = pathlib.Path(file).expanduser().resolve()
        out = saver(f, *args, **kwargs)
        os.chmod(f, mode=0o770)
        os.chown(f, uid=-1, gid=os.stat(f.parent).st_gid)
        return out

    return wrapper


def fix_perms(filepath: str | pathlib.Path | gzip.GzipFile | io.FileIO):
    """takes file and sets 770 permissions and group ownership"""

    try:
        try:
            f = pathlib.Path(filepath).expanduser().resolve()
        except TypeError:
            f = pathlib.Path(filepath.name).expanduser().resolve()
        os.chmod(f, mode=0o770)
        os.chown(f, uid=-1, gid=os.stat(f.parent).st_gid)
        return f
    finally:
        warnings.warn(f"Fixing permissions for {filepath=} failed.")
