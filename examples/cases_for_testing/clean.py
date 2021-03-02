import os
from util import util
import shutil


def clean():
    root = "./"
    cases = util.get_all_the_dirs_with_filter(root)
    reserve_files = ["TeethEdgeDownNew.png", "TeethEdgeUpNew.png"]
    for c in cases:
        path = os.path.join(c, "steps")
        steps = os.listdir(path)
        for s in steps:
            spath = os.path.join(path, s)
            if s.startswith("step"):
                files = os.listdir(spath)
                for f in files:
                    if f not in reserve_files:
                        util.remove_files(os.path.join(spath, f))
            else:
                shutil.rmtree(spath)

        if os.path.exists(os.path.join(c, "results")):
            shutil.rmtree(os.path.join(c, "results"))


if __name__ == "__main__":
    clean()

