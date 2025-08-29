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


def set_perms(saver: typing.Callable, compress: bool = False) -> typing.Callable:
    """sets 770 permissions and group ownership to that of the enclosing folder during save"""

    @functools.wraps(saver)
    def wrapper(file, *args, **kwargs):
        f = pathlib.Path(file).expanduser().resolve()
        if compress:
            with gzip.open(f, "wb") as fgz:
                out = saver(fgz, *args, **kwargs)
        else:
            out = saver(f, *args, **kwargs)
        os.chmod(f, mode=0o770)
        os.chown(f, uid=-1, gid=os.stat(f.parent).st_gid)
        return out

    return wrapper


def fix_perms(filepath: str | pathlib.Path | gzip.GzipFile | io.FileIO) -> pathlib.Path:
    """takes file and sets 770 permissions and group ownership"""

    try:
        try:
            f = pathlib.Path(filepath).expanduser().resolve()
        except TypeError:
            f = pathlib.Path(filepath.name).expanduser().resolve()
        os.chmod(f, mode=0o770)
        os.chown(f, uid=-1, gid=os.stat(f.parent).st_gid)
        return f
    except Exception as e:
        warnings.warn(f"Fixing permissions for {filepath=} failed due to {e}.")


if __name__ == "__main__":
    import subprocess
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        set_perms(np.save, compress=True)(tmpdir + "/test.npy.gz", np.arange(1000))
        np.save(f := (tmpdir + "/test2.npy"), np.arange(100))
        fix_perms(f)
        result = subprocess.run(["ls", "-lt"], capture_output=True, text=True)
        print(result.stdout)
