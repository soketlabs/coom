from nemo.collections.llm import PreTrainingDataModule
import nemo.collections.llm.gpt.data.pre_training as pre_training

def eka_validate_dataset_asset_accessibility(paths):
    """
    Validate the accessibility of the dataset assets.
    """
    if paths is None:
        raise ValueError("Expected path to have a value.")

    if isinstance(paths, tuple) or isinstance(paths, list):
        if pre_training.is_zipped_list(paths):
            # remove weights from paths.
            paths = paths[1::2]
        for p in paths:
            eka_validate_dataset_asset_accessibility(p)
        return
    elif isinstance(paths, dict):
        for p in paths.values():
            eka_validate_dataset_asset_accessibility(p)
        return

    if not isinstance(paths, str) and not isinstance(paths, Path):
        raise ValueError("Expected path to be of string or Path type.")

    suffices = (".bin", ".idx")
    if pre_training.is_multistorageclient_url(paths):
        try:
            for suffix in suffices:
                file_path = (paths + suffix)
                msc = pre_training.import_multistorageclient()
                path = msc.Path(file_path)
            return
        except:
            raise FileNotFoundError(f"Expected {str(file_path)} to exist in the msc bucket..")
    else:
        path = Path(paths)


    if path.is_dir():
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Expected {str(path)} to be readable.")
        # Will let the downstream class confirm contents are ok.
        return
    if path.exists():
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Expected {str(path)} to be readable.")
        return
    for suffix in suffices:
        file_path = path.with_name(path.name + suffix)
        if not file_path.exists():
            raise FileNotFoundError(f"Expected {str(file_path)} to exist.")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Expected {str(file_path)} to be readable.")

pre_training.validate_dataset_asset_accessibility =  eka_validate_dataset_asset_accessibility

class EKAPreTrainingDataModule(PreTrainingDataModule):
    """
    Currently identical to NeMo's PreTrainingDataModule.
    Defined separately for modularity, to allow future changes or extensions.
    """
    pass

