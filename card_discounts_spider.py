import math
import os
import json
import re
import logging
import time
from io import BytesIO
from threading import Lock
from multiprocessing import get_context
from multiprocessing.queues import Queue

import cv2
import parsel
import requests
import pytesseract
import numpy as np
from PIL import Image
from cnocr import CnOcr
from pyzbar import pyzbar
from paddleocr import PaddleOCR, paddleocr
from fake_useragent import UserAgent
from pyzbar.wrapper import ZBarSymbol
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)


def get_img_limit_len(height):
    limit_multiplier = 32
    return limit_multiplier * (np.ceil(height / limit_multiplier))


class CardDiscountsSpider:
    def __init__(self):
        self.ua = UserAgent()
        self.base_path = r'.\banks_discounts_data'
        self.bank_api = 'https://point.95516.com/quanyiinfo/server/files/json/mobile/bankZone/bsOa.json'
        self.discounts_url = 'https://point.95516.com/quanyiinfo/server/up-interest/filterSearch'
        self.discount_api = 'https://point.95516.com/quanyiinfo/server/files/json/{id}/{id}.json'
        # self.discount_backup_api = 'https://point.95516.com/quanyiinfo/server/files/json/{id}/{id}_1_10.json'
        self.banks_queue = Queue(ctx=get_context())
        self.discounts_queue = Queue(ctx=get_context())
        self.lock = Lock()
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False,
                                    enable_mkldnn=True, use_tensorrt=True, use_mp=True,
                                    det_limit_side_len=8000)

    def handle_request(self, url, data=None, method='GET', referer='', **kwargs):
        """
        发送请求.
        :param url: 请求的链接
        :param data: 请求携带的数据，适用于 POST 请求
        :param method: 请求方式
        :param referer: 请求头中的携带的请求源
        :param kwargs: 发送请求携带的参数，type: dict, 将会与请求头合并
        :return: 响应对象
        """
        headers = {
                      'Content-Type': 'application/json;charset=utf-8',
                      'User-Agent': self.ua.random,
                      'finger-id': '4658c136e25408c940bf87f01dd3afc1',
                      'Referer': referer,
                      'Host': 'point.95516.com',
                      'Origin': 'https://point.95516.com',
                  } | kwargs
        if data:
            return requests.request(method=method, url=url, headers=headers, data=json.dumps(data))
        return requests.request(method=method, url=url, headers=headers)

    def handle_banks_list(self):
        """ 获取优惠中心所有银行信息，并添加到队列中. """
        referer = 'https://point.95516.com/quanyiinfo/'
        res = self.handle_request(self.bank_api, referer=referer).json()
        if res['message'] == '成功':
            for item in res['data'][0]['upBanks']:
                bank_id = item["bankId"]
                bank_data = {
                    'bank_name': item['bankName'],
                    'bank_id': bank_id,
                    'bank_link': f'https://point.95516.com/quanyiinfo/entitlement?bankId={bank_id}',
                    'discounts_count': math.ceil(item['interestCount'] / 10)  # 获取页数
                }
                self.banks_queue.put(bank_data)

    def handle_discounts(self, bank_data):
        """
        通过银行数据构造请求，获取每个银行的所有优惠信息，构造每个优惠的api接口并添加到队列中.
        :param bank_data: 银行信息数据
        """
        for page in range(1, bank_data['discounts_count'] + 1):
            print(f"{bank_data['bank_name']},第{page}页 共{bank_data['discounts_count']}页")
            data = {
                "bankCodes": [
                    bank_data['bank_id']
                ],
                "themeId": [],
                "cardType": [],
                "bankCardLvl": [],
                "cityId": "bsOa",
                "provinceId": "bsOa",
                "currentPage": page
            }
            res = self.handle_request(url=self.discounts_url,
                                      data=data,
                                      method='POST',
                                      referer=bank_data['bank_link']).json()
            if res['message'] == '成功':
                for item in res['data']:
                    discount_id = item['id']
                    discount_data = {
                        'discount_name': item['interestShowName'],
                        'discount_link': self.discount_api.format(id=discount_id),
                        'id': discount_id
                    }
                    # 位或运算赋值, 合并两个字典
                    discount_data |= bank_data
                    self.discounts_queue.put(discount_data)

    def get_discount(self):
        """ 根据优惠api获取优惠数据 """
        # 添加线程锁
        self.lock.acquire()
        discounts_data = self.discounts_queue.get()
        print(f'正在处理{discounts_data["discount_name"]} 优惠数据')
        res = self.handle_request(url=discounts_data['discount_link']).json()
        if res['message'] == '成功' and res['data'][0]['content']:
            self.lock.release()
            self._discount_extracted_save_dir(res, discounts_data)
        else:
            self.lock.release()
            print("错误数据链接", discounts_data['discount_link'])
        self.lock.release()

    def _discount_extracted_save_dir(self, res, discounts_data):
        """
        从响应中提取数据，并保存
        :param res: 响应对象
        :param discounts_data: 优惠信息数据
        """
        discount_name = res['data'][0]['interestShowName']

        pattern = re.compile(r'[\?\,\/\_\*\<\>\|\:]')

        content, qr_code_url = self.get_content(res['data'][0]['content'])
        if not content:
            print('content为空', res['data'][0]['id'])
        data = {
            'discount_name': re.sub(pattern, '-', discount_name),
            'equity_rules': {
                "activity_time": res['data'][0].get('activityTime', None),
                "activeObject": res['data'][0].get('activeObject', None),
                "limitNumber": res['data'][0].get('limitNumber', None),
                'content': content,
                'qr_code_url': qr_code_url,
            }
        }
        print('正在保存数据')
        self.save_to_directory(discounts_data, data)

    def get_content(self, content):
        """
        提前content中的数据，如果包含图片则识别图片中数据.
        :param content: 为html标签字符串
        :returns: 识别图片中的文本. 图片中二维码的链接
        """
        sel = parsel.Selector(content)
        active_rules = sel.css('p')[0]
        if (p_len := len(p := sel.css('p'))) >= 3:
            active_rules = p[:-2]
        elif p_len < 3:
            active_rules = p[:-1]

        # 对图片进行识别
        img_text, qr_code_url = '', []
        if img_urls := active_rules.css('img::attr(src)').getall():
            for img_url in img_urls:
                img_result = self.get_img_data(img_url)
                img_text += f'-{img_result[0]}'
                if url := img_result[1]:
                    qr_code_url += url

        return ''.join(active_rules.css('::text').getall()) + img_text, qr_code_url

    def get_img_data(self, img_url):
        """
        识别图片中的数据.
        :param img_url: 图片链接
        :returns: 图片中的文本。 图片中二维码的链接
        """
        print(f'正在对图像进行检测识别: {img_url}')
        res = self.handle_request(url=img_url,
                                  Accept='image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8').content

        # 二进制转为矩阵
        image = np.asarray(bytearray(res), dtype="uint8")
        img_array = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # easyocr
        # 图片类型： file_path or numpy-array or byte stream  精准不是很高，速度因为没有GPU， 稍慢, 提速需要下载pytorch 1.2G
        # result = easyocr.Reader(['ch_sim']).readtext(res)
        # ','.join([i[1] for i in result])

        # PaddleOCR
        # 图片类型： ndarray or img_path   较高，稍快
        result = self.paddle_ocr.ocr(img_array, cls=True)
        # ','.join([i[1][0] for i in result[0]])

        # pytesseract
        # 图片类型： image obj 精准度最差
        # result = pytesseract.image_to_string(Image.open(BytesIO(res)), lang='chi_sim')

        # CnOcr
        # 图片类型： 二进制矩阵 精度较高， 速度较快
        # result_4 = CnOcr().ocr(img_array)
        # ','.join([i['text'] for i in result_4])

        return (
            ','.join([i[1][0] for i in result[0]]),
            [rq_code.data.decode('utf-8') for rq_code in rq_codes]
            if (rq_codes := pyzbar.decode(img_array, symbols=[ZBarSymbol.QRCODE]))
            else None
        )

    def save_to_directory(self, discounts_data, data):
        """
        将数据保存到对应文件夹中.
        :param discounts_data: 优惠数据信息，获取优惠银行、优惠名称
        :param data: 保存的数据
        """
        if not os.path.exists(file_path := os.path.join(self.base_path, discounts_data['bank_name'])):
            os.makedirs(file_path)
        print(f'正在保存{file_path} 路径下，优惠名称：{data["discount_name"]}')
        # 以id命名防止重名
        with open(f"{file_path}\\{data['discount_name']}.json", mode='a+', encoding='utf-8') as f:
            # 不对中文编码
            f.write(json.dumps(dict(data), ensure_ascii=False))
            f.write('\n')


if __name__ == '__main__':
    spider = CardDiscountsSpider()
    spider.handle_banks_list()

    while spider.banks_queue.qsize() > 0:
        time.sleep(0.2)
        spider.handle_discounts(spider.banks_queue.get())

    print(f'共有{spider.discounts_queue.qsize()}个优惠================================================')
    while spider.discounts_queue.qsize() > 0:
        spider.get_discount()
    #
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     while spider.discounts_queue.qsize() > 0:
    #         executor.submit(spider.get_discount)
