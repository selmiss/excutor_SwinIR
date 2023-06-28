import os.path

from jina import Flow, Document, DocumentArray, Client
from PIL import Image
import io

FLOW_CONFIG_PATH = 'flow.yml'
# ENDPOINT = None
ENDPOINT = 'grpc://0.0.0.0:51000'
# ENDPOINT = "grpcs://inspired-mallard-b41d2134e0-grpc.wolf.jina.ai"

IMAGES = [
    'https://replicate.delivery/mgxm/efd1b6b0-4d79-4a42-ab31-2dcd29754a2d/chip.png'
]

image_da = DocumentArray([Document(uri=uri, tags={'uri': uri}).load_uri_to_blob() for uri in IMAGES])


if ENDPOINT:
    client = Client(host=ENDPOINT)

    print(f'===> Super-resolution task ...')
    result = client.post(on='/upscale', inputs=image_da, parameters={'output_format': 'blob'}, show_progress=True)
    if not os.path.exists('results'):
        os.mkdir('results')
    for idx, r in enumerate(result):
        print(f'==> {r.summary()}')
        pil_img = Image.open(io.BytesIO(r.blob)).convert('RGB')
        print(f"output size: {pil_img.size}")
        print('inference run time: {}s'.format(r.tags['runtime']))
        print(f"output height: {pil_img.height}, output width: {pil_img.width}")
        pil_img.save(f"./results/{str(idx)}.png", format='PNG', quality=100)

else:
    with Flow().load_config(FLOW_CONFIG_PATH) as f:
        print(f'===> Super-resolution task ...')
        result = f.post(
            '/upscale', inputs=image_da, show_progress=True
        )
        if not os.path.exists('results'):
            os.mkdir('results')
        for idx, r in enumerate(result):
            print(f'==> {r.summary()}')
            pil_img = Image.open(io.BytesIO(r.blob)).convert('RGB')
            print(f"output size: {pil_img.size}")
            print('inference run time: {}s'.format(r.tags['runtime']))
            print(f"output height: {pil_img.height}, output width: {pil_img.width}")
            pil_img.save(f"./results/{str(idx)}.png", format='PNG', quality=100)
