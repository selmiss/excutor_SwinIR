from typing import Tuple, Dict, Optional
from jina import DocumentArray, Executor, requests
import torch
from jina.logging.logger import JinaLogger
import os
import time
import numpy as np
from multiprocessing.pool import ThreadPool
from .SwinIR.models.network_swinir import SwinIR as net


S3_PATH = (
    'https://clip-as-service.s3.us-east-2.amazonaws.com/models/super_resolution/swin_ir'
)
TMP_PATH = './tmp'


class SwinIRExecutor(Executor):
    """"""
    def __init__(
            self,
            model_name: str = 'real_sr::BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN',
            minibatch_size: int = 32,
            num_worker_preprocess: int = 4,
            upscale: int = 4,
            large_model: bool = False,
            tile: int = None,
            tile_overlap: int = 32,
            device: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.upscale = upscale
        self.large_model = large_model
        self.tile = tile
        self.tile_overlap = tile_overlap

        self.model_name = model_name
        self.model_type, self.s3_file_name = self.model_name.split('::')
        self.set_max_size()

        model_path = os.path.join(TMP_PATH, self.s3_file_name + '.pth')

        if not os.path.exists(model_path):
            self.download_model(
                os.path.join(S3_PATH, self.s3_file_name + '.pth'),
                os.path.join(TMP_PATH, self.s3_file_name + '.pth'),
            )

        self._minibatch_size = minibatch_size

        self._pool = ThreadPool(processes=num_worker_preprocess)

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device

        self._init_model_weights()

    def _init_model_weights(self):
        self.logger.info(
            f"Model initialization start, model_name: {self.model_name}"
        )
        if self.model_type == 'classical_sr':
            model = net(upscale=self.upscale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            param_key_g = 'params'

        # 002 lightweight image sr
        # use 'pixelshuffledirect' to save parameters
        elif self.model_type == 'lightweight_sr':
            model = net(upscale=self.upscale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
            param_key_g = 'params'

        # 003 real-world image sr
        elif self.model_type == 'real_sr':
            if not self.large_model:
                # use 'nearest+conv' to avoid block artifacts
                model = net(upscale=self.upscale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            else:
                # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
                model = net(upscale=self.upscale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key_g = 'params_ema'

        pretrained_model = torch.load(os.path.join(TMP_PATH, self.s3_file_name + '.pth'))
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        model.eval()
        model = model.to(self._device)
        self.model = model

        self.logger.info(
            f"Model initialization success!"
        )

    def set_max_size(self):
        self.max_size = 1024 * 1024

    def preproc_image(
            self,
            da: 'DocumentArray',
            drop_image_content: bool = False,
    ) -> Tuple['DocumentArray', torch.tensor]:

        tensors_batch = []

        for d in da:
            if d.blob:
                d.convert_blob_to_image_tensor()
            elif d.uri:
                d.load_uri_to_image_tensor()
            image = d.tensor
            if image is None:
                raise ValueError(f"input image is None")
            tensors_batch.append(image)

            if drop_image_content:
                d.pop('blob', 'tensor')

        return da, tensors_batch

    def sr_inference(self, img_lq, window_size):
        start_time = time.perf_counter()
        if self.tile is None:
            # test the image as a whole
            output = self.model(img_lq)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(self.tile, h, w)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = self.tile_overlap
            sf = self.upscale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)
        end_time = time.perf_counter()
        return output, round(end_time - start_time, 3)

    @requests(on='/upscale')
    def upscale(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):

        _output_format = parameters.get('output_format', 'tensor')
        if _output_format not in ['tensor', 'blob']:
            raise ValueError(
                f"output format only supports tensor and blob, got {_output_format}"
            )

        # docs, _ = self.preproc_image(docs)
        for idx, d in enumerate(docs):
            if d.blob:
                d.convert_blob_to_image_tensor()
            assert d.tensor is not None, "Failed to load image."

            if (
                    self.max_size
                    and d.tensor.shape[0] * d.tensor.shape[1] > self.max_size
            ):
                raise ValueError(
                    f"Max image pixels for input is {self.max_size} "
                    f"(height * width), but got {d.tensor.shape[0] * d.tensor.shape[1]} "
                    f"({d.tensor.shape[0]} * {d.tensor.shape[1]})"
                )

            img_lq = d.tensor.astype(np.float32) / 255.
            img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self._device)  # CHW-RGB to NCHW-RGB

            # inference
            with torch.no_grad():
                window_size = 8
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output, runtime = self.sr_inference(img_lq, window_size)
                output = output[..., :h_old * self.upscale, :w_old * self.upscale]

            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            d.tags = {'runtime': runtime}
            d.tensor = output
            if _output_format == 'blob':
                d.convert_image_tensor_to_blob()

        return docs

    @staticmethod
    def download_model(url, dst, hash_prefix=None, progress=True):
        r"""Download object at the given URL to a local path.
            Refer to torch.hub.download_url_to_file
        Args:
            url (str): URL of the object to download
            dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
            hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
                Default: None
            progress (bool, optional): whether or not to display a progress bar to stderr
                Default: True
        """
        from urllib.request import urlopen, Request
        import shutil
        from tqdm import tqdm
        import hashlib
        import tempfile

        file_size = None
        req = Request(url, headers={"User-Agent": "torch.hub"})
        u = urlopen(req)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

        # We deliberately save it in a temp file and move it after
        # download is complete. This prevents a local working checkpoint
        # being overridden by a broken download.
        dst = os.path.expanduser(dst)
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

        try:
            if hash_prefix is not None:
                sha256 = hashlib.sha256()
            with tqdm(
                    total=file_size,
                    disable=not progress,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if hash_prefix is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

            f.close()
            if hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[: len(hash_prefix)] != hash_prefix:
                    raise RuntimeError(
                        'invalid hash value (expected "{}", got "{}")'.format(
                            hash_prefix, digest
                        )
                    )
            shutil.move(f.name, dst)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)
