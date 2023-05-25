import logging
import torch
TMP_PATH = './checkpoints'


def load_state_dict(
        target_path: str=None,
        file_name: str = None,
        url: str = None,
):
    if file_name is None and url is None:
        raise ValueError(f"you must pass either file_name or url to load model")

    from pathlib import Path

    tmp_path = Path(target_path)
    tmp_path.mkdir(exist_ok=True)

    file_name = url.split('/')[-1]
    if not (tmp_path / file_name).is_file():
        logging.info(f"===> downloading from: {url}")
        torch.hub.download_url_to_file(url, str(tmp_path / file_name))
        logging.info(f"==> download done, saved to: {str(tmp_path / file_name)}")
