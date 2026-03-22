def get_bulkdata_dir():
    from importlib.resources import files
    from pathlib import Path

    return Path(files("neuromechants")).parent.parent / "bulkdata"
