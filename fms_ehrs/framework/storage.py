#!/usr/bin/env python3

"""
provides functionality related to saving data and artifacts
"""

import functools
import os
import pathlib


def fix_perms(saver):
    """sets 770 permissions and group ownership to that of the enclosing folder"""

    @functools.wraps(saver)
    def wrapper(file, *args, **kwargs):
        f = pathlib.Path(file).expanduser().resolve()
        out = saver(f, *args, **kwargs)
        os.chmod(f, mode=0o770)
        os.chown(f, uid=-1, gid=os.stat(f.parent).st_gid)
        return out

    return wrapper
