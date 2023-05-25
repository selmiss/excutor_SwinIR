from typing import Tuple, Dict, Union, Optional
from jina import DocumentArray, Executor, requests
import torch
from PIL import Image
from jina.logging.logger import JinaLogger
import requests as requests_ori
import os
import io
import numpy as np
from multiprocessing.pool import ThreadPool
from .SwinIR.models.network_swinir import SwinIR as net


class SRExecutor(Executor):
    """"""
    def __init__(
            self,
            model_name: str = 'real_sr::003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN',
            minibatch_size: int = 32,
            num_worker_preprocess: int = 4,
            upscale: int = 4,
            large_model: bool = False,
            training_patch_size: int = 128,
            tile: int = None,
            tile_overlap: int = 32,
            device: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.upscale = upscale
        self.large_model = large_model
        self.training_patch_size = training_patch_size
        self.tile = tile
        self.tile_overlap = tile_overlap

        self.model_name, self.pretrained = model_name.split('::')
        self.pretrained = './checkpoints/' + self.pretrained + '.pth'

        if os.path.exists(self.pretrained):
            print(f'loading model from {self.pretrained}')
        else:
            os.makedirs(os.path.dirname(self.pretrained), exist_ok=True)
            url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(self.pretrained))
            r = requests_ori.get(url, allow_redirects=True)
            print(f'downloading model {self.pretrained}')
            open(self.pretrained, 'wb').write(r.content)
        self._minibatch_size = minibatch_size

        self._pool = ThreadPool(processes=num_worker_preprocess)

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device

        self._init_model_weights()

    def _init_model_weights(self):
        self.logger.info(
            f"Model initialization start, model_name: {self.model_name}, "
            f"pretrained: {self.pretrained}..."
        )
        if self.model_name == 'classical_sr':
            model = net(upscale=self.upscale, in_chans=3, img_size=self.training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            param_key_g = 'params'

        # 002 lightweight image sr
        # use 'pixelshuffledirect' to save parameters
        elif self.model_name == 'lightweight_sr':
            model = net(upscale=self.upscale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
            param_key_g = 'params'

        # 003 real-world image sr
        elif self.model_name == 'real_sr':
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

        # 004 grayscale image denoising
        elif self.model_name == 'gray_dn':
            model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 005 color image denoising
        elif self.model_name == 'color_dn':
            model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 006 grayscale JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif self.model_name == 'jpeg_car':
            model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 006 color JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif self.model_name == 'color_jpeg_car':
            model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        pretrained_model = torch.load(self.pretrained)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        model.eval()
        model = model.to(self._device)
        self.model = model

        self.logger.info(
            f"Model initialization success!"
        )

    def preproc_image(
            self,
            da: 'DocumentArray',
            device: str = 'cpu',
            drop_image_content: bool = False,
    ) -> Tuple['DocumentArray', torch.tensor]:

        tensors_batch = []

        for d in da:
            # content = d.content
            if d.blob:
                image = Image.open(io.BytesIO(d.blob)).convert('RGB')
                d.convert_blob_to_tensor()
            elif d.uri:
                # image = Image.open(io.BytesIO(d.load_uri_to_blob().blob)).convert('RGB')
                image = d.load_uri_to_image_tensor().tensor
            elif d.tensor is not None:
                # image = Image.fromarray(d.tensor).convert('RGB')
                image = d.tensor
            tensors_batch.append(image)

            # recover doc content
            # d.content = content
            if drop_image_content:
                d.pop('blob', 'tensor')

        return da, tensors_batch

    def sr_inference(self, img_lq, window_size):
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
        return output

    @requests(on='/sr')
    def sr(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        docs, _ = self.preproc_image(docs)
        for idx, d in enumerate(docs):
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
                output = self.sr_inference(img_lq, window_size)
                output = output[..., :h_old * self.upscale, :w_old * self.upscale]

            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            d.embedding = output
        return docs


'''
from jina import Deployment

with Deployment(uses=SRExecutor, port=12345, replicas=2) as dep:
    print('SRmodel started...')
    dep.block()
'''