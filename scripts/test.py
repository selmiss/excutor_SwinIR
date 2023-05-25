from jina import Client, Document, DocumentArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def sr_test():
    image_uri = './SwinIR/testsets/RealSRSet+5images/chip.png'
    img = mpimg.imread(image_uri)

    response = Client(host='grpc://127.0.0.1:51000').post(
        on='/sr',  inputs=DocumentArray([Document(uri=image_uri)]), show_progress=True
    )
    result = response[0].embedding
    print('ori', type(img), img.shape)
    print('result', type(result), result.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(result)
    plt.show()


if __name__ == '__main__':
    sr_test()

