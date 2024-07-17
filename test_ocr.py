from io import BytesIO
from time import process_time, perf_counter

import cv2
import easyocr
import requests
import numpy as np
import pytesseract
from PIL import Image
from cnocr import CnOcr
from cnstd import CnStd
from pyzbar import pyzbar
from pyzbar.wrapper import ZBarSymbol
from paddleocr import PaddleOCR, draw_ocr


def get_img_limit_len(height):
    limit_multiplier = 32
    return limit_multiplier * (np.ceil(height / limit_multiplier))


def get_img_data(img_url):
    """ 识别图片中的数据 """
    res = requests.get(url=img_url).content
    image_obj = np.asarray(bytearray(res), dtype="uint8")  # 二进制转为矩阵
    img_array = cv2.imdecode(image_obj, cv2.IMREAD_COLOR)

    # 需要 ndarray or img_path
    r1_s = perf_counter()
    # ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    # result = ocr.ocr(img_array)
    # print(','.join([i[1][0] for i in result[0]]))

    # image = Image.open(BytesIO(res)).convert('RGB')
    # boxes = [line[0] for line in result[0]]
    # txts = [line[1][0] for line in result[0]]
    # scores = [line[1][1] for line in result[0]]
    # im_show = draw_ocr(image, boxes, txts, scores)
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')
    r1_e = perf_counter()
    print("PaddleOCR 运行时间是: {:.3f}s".format(r1_e - r1_s))
    print()

    r5_s = perf_counter()
    ocr_2 = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, enable_mkldnn=True,
                      use_tensorrt=True, use_mp=True, det_limit_side_len=get_img_limit_len(img_array.shape[0]))
    result_5 = ocr_2.ocr(img_array)
    print(','.join([i[1][0] for i in result_5[0]]))
    r5_e = perf_counter()
    print("PaddleOCR 优化后的运行时间是: {:.3f}s".format(r5_e - r5_s))
    print()

    # 可需要 file_path or numpy-array or byte stream
    r2_s = perf_counter()
    result_2 = easyocr.Reader(['ch_sim']).readtext(res)
    print(','.join([i[1] for i in result_2]))
    r2_e = perf_counter()
    print("easyocr 运行时间是: {:.3f}s".format(r2_e - r2_s))
    print()

    r3_s = perf_counter()
    result_3 = pytesseract.image_to_string(Image.open(BytesIO(res)), lang='chi_sim')
    print(result_3)
    r3_e = perf_counter()
    print("pytesseract 运行时间是: {:.3f}s".format(r3_e - r3_s))
    print()

    r4_s = perf_counter()
    result_4 = CnOcr().ocr(img_array)
    print(','.join([i['text'] for i in result_4]))
    r4_e = perf_counter()
    print("CnOcr 运行时间是: {:.3f}s".format(r4_e - r4_s))

    # box_info_list = CnStd().detect(img_array)  # 检测文本
    #
    # return (
    #         ','.join([i[1][0] for i in result[0]]),
    #         [rq_code.data.decode('utf-8') for rq_code in rq_codes]
    #         if (rq_codes := pyzbar.decode(img_array, symbols=[ZBarSymbol.QRCODE]))
    #         else None
    #     )


if __name__ == '__main__':
    url = 'https://point.95516.com/quanyiinfo/files/img/a8af90a4b038542e38ca84d44c669846.png'
    url2 = 'https://point.95516.com/quanyiinfo/files/img/1a91fec03dc3dc96a20271013583a7f1.png'
    url3 = 'https://point.95516.com/quanyiinfo/files/img/b43b959fb6c034aac2d727045e41ecb6.jpg'
    url4 = 'https://point.95516.com/quanyiinfo/files/img/554e87e9f9c25507bdcd952ab6c976e8.jpg'

    url5 = 'https://point.95516.com/quanyiinfo/files/img/2697b4f1c6e8cd38f3b821cbb98245f9.jpg'
    url6 = 'https://point.95516.com/quanyiinfo/files/img/db4c2275ea020be44b30713641719196.jpg'
    url7 = 'https://point.95516.com/quanyiinfo/files/img/57c0629686da730da6ed697646b77cb9.jpg'
    # print(get_img_data(url5))
    get_img_data(url5)
