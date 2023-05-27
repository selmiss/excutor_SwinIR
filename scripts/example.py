from jina import Flow, Document, DocumentArray, Client
from PIL import Image
import io

FLOW_CONFIG_PATH = 'flow.yml'
ENDPOINT = None
# ENDPOINT = 'grpc://0.0.0.0:51000'
# ENDPOINT = "grpcs://inspired-mallard-b41d2134e0-grpc.wolf.jina.ai"

IMAGES = [
    'https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0rw369i5h9t%2Foriginal.png',
    'https://storage.googleapis.com/causal-diffusion.appspot.com/imagePrompts%2F0ixde4g8gwte%2Foriginal.png',
    'https://clip-as-service.s3.us-east-2.amazonaws.com/models/super_resolution/cv2/Best-Honey-Butter-Roasted-Carrots0-1.jpg',
]

image_da = DocumentArray([Document(uri=uri, tags={'uri': uri}).load_uri_to_blob() for uri in IMAGES])


if ENDPOINT:
    client = Client(host=ENDPOINT)

    print(f'===> Super-resolution task ...')
    result = client.post(on='/upscale', inputs=image_da, show_progress=True)
    for idx, r in enumerate(result):
        print(f'==> {r.summary()}')
        pil_img = Image.open(io.BytesIO(r.blob)).convert('RGB')
        print(f"output size: {pil_img.size}")
        print(f"output height: {pil_img.height}, output width: {pil_img.width}")
        pil_img.save(f"./{str(idx)}.png", format='PNG', quality=100)

else:
    with Flow().load_config(FLOW_CONFIG_PATH) as f:
        print(f'===> Super-resolution task ...')
        result = f.post(
            '/upscale', inputs=image_da, show_progress=True
        )
        for idx, r in enumerate(result):
            print(f'==> {r.summary()}')
            pil_img = Image.open(io.BytesIO(r.blob)).convert('RGB')
            print(f"output size: {pil_img.size}")
            print(f"output height: {pil_img.height}, output width: {pil_img.width}")
            pil_img.save(f"./{str(idx)}.png", format='PNG', quality=100)
