import os


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"creating path {path}")
        return
    print(f"check path exists {path}")
