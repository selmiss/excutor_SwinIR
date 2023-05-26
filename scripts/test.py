from jina import Client, Document, DocumentArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


def sr_test():
    image_uri = './SwinIR/testsets/RealSRSet+5images/chip.png'
    img = mpimg.imread(image_uri)
    start_time = time.perf_counter()
    response = Client(host='grpc://127.0.0.1:50635').post(
        on='/sr',  inputs=DocumentArray([Document(uri=image_uri)]), show_progress=True
    )
    end_time = time.perf_counter()

    result = response[0].embedding
    print('ori', type(img), img.shape)
    print('result', type(result), result.shape)
    print('inference run time: {}s'.format(response[0].tags['runtime']))
    print('response time: {}s'.format(round((end_time - start_time), 3)))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(result)
    plt.show()


if __name__ == '__main__':
    sr_test()

