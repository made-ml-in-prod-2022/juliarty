import os


def wait_for_data_func(
    execution_time: str, data_path_format: str, target_path_format: str
):
    """
    Args:
        execution_time: execution datetime
    """
    return os.path.exists(data_path_format.format(execution_time)) and (
        target_path_format is None
        or os.path.exists(target_path_format.format(execution_time))
    )
