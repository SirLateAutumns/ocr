# 图片识别分享

1. OCR简介

   OCR的全称叫作“Optical Character Recognition”，即光学字符识别。

   OCR作用是对包含文本资料的图像文件进行分析识别处理，获取文字及版面信息，主要针对文字比较规范，没什么底色或底色单一的图片。

2. STD简介

   场景文字识别（scene text recognition），简称为STR

   STR针对的图片是更一般的复杂场景，比如随手拍的照片中广告牌上的文字，店铺名称等。

## 一、工具

### 1. EasyOCR

1. 介绍

   [EasyOCR](https://github.com/JaidedAI/EasyOCR) 是一个用于从图像中提取文本的 python 模块, 它是一种通用的 OCR，既可以读取自然场景文本，也可以读取文档中的密集文本。目前支持 80 多种语言和所有流行的书写脚本，包括：拉丁文、中文、阿拉伯文、梵文、西里尔文等。安装

   ```python
   pip install easyocr
   ```

3. 使用

   ```python
   import easyocr
   # 创建reader对象
   reader = easyocr.Reader(lang_list=['ch_sim','en']) 
   # 读取图像
   result = reader.readtext(image='test.jpg')
   # 输出结果
   print(','.join([i[1] for i in result]))
   ```

4. 参数:

   ​	lang_list: 传入需要识别的语言，是一个列表.

   ​	image: 可传入图像路径、numpy数组、字节流对象

5. 注意:

   ​	1. 在第一次安装使用时，easyocr会自动下载默认模型。默认路径为：C:\Users\用户名.EasyOCR\model，可通	`model_storage_directory`改变	模型存放路径。如果下载慢，可通过[官网](https://www.jaided.ai/easyocr/modelhub/)下载，分别为zh_sim_g2(中文简体)、english_g2(英	文)，放在默认路径下即可。

   ​	2. easyocr模型在GPU上运行效率更高，则需要下载[PyTorch](https://pytorch.org/)与CUDA。

   ​	3. 如果显卡为AMD显卡，将不支持CUDA，需要下载CPU版本的pytorch，英伟达显卡选择CUDA版本，版本号需要根据自己电脑	英伟达版本决定，cmd使用`nvidia-smi`查看CUDA版本。[CUDA 12.0 Release Notes (nvidia.com)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)查看对应版本限制。

### 2. PaddleOCR

1. 介绍

   [PaddleOCR (gitee.com)](https://gitee.com/paddlepaddle/PaddleOCR)基于Paddle的OCR工具库，包含总模型仅8.6M的超轻量级中文OCR，单模型支持中英文数字组合识别、竖排文本识别、长文本识别。同时支持多种文本检测、文本识别的训练算法。旨在打造一套丰富、领先、且实用的OCR工具库，助力开发者训练出更好的模型，并应用落地。

2. 安装

   首先安装PaddlePaddle、paddleocr。

   ```
   pip install paddlepaddle paddleocr>=2.0.1 shapely -i https://mirror.baidu.com/pypi/simple
   ```

   对于win如果在安装过程中有 **shapely** 库安装失败，需要在[Archived: Python Extension Packages for Windows - Christoph Gohlke (uci.edu)](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)下载whl重新安装。

3. 使用

   ```python
   import cv2
   import numpy as np
   from paddleocr import PaddleOCR
   
   # 创建paddle对象
   ocr = PaddleOCR(use_angle_cls=True, lang="ch")
   # 准备图像
   img_path = "./4A34A16F.png"
   image = np.asarray(bytearray(res), dtype="uint8")  # 二进制转为矩阵
   img_array = cv2.imdecode(image, cv2.IMREAD_COLOR)
   # 输出结果保存路径
   result = ocr.ocr(img=img_path, det=True, rec=True)
   # 打印结果
   print(','.join([i[1][0] for i in result[0]]))注意：
   ```

4. 参数：

   use_angle_cls: 用于确定是否使用角度分类模型，即是否识别垂直方向的文字.

   lang: 表示识别的语言.

   img: 图像路径、ndarray数组

   cls: 表示识别的语言.默认为True。

   rec: 是否使用文本识别.默认为True。

   det: 是否使用文本检测.默认为True。

5. 注意：

    在第一次安装使用时，easyocr会自动下载默认模型。默认路径为：C:\Users\用户名\\.paddleocr\whl\[cls、det、rec]\[ch、]\\...，分别有三个目录。在[gitee.com](https://gitee.com/paddlepaddle/PaddleOCR#https://gitee.com/link)中的系列模型中下载，需要解压后分别放到默认路径下。

### 3.PyTesseract

1. 介绍

   [pytesseract: A Python wrapper for Google Tesseract](https://github.com/madmaze/pytesseract)是基于Python的OCR工具， 底层使用的是Google的Tesseract-OCR 引擎，支持识别图片中的文字，支持jpeg, png, gif, bmp, tiff等图片格式。

2. 安装

   2.1 首先需要下载Tesseract应用程序

   ​		Windows Tesseract下载地址：https://digi.bib.uni-mannheim.de/tesseract/

   ​		Mac和Linux安装方法参考：https://tesseract-ocr.github.io/tessdoc/Installation.html

   ​	安装时可以选择需要的语言包，在这里选择下载中文语音包（chi_sim）：

   ![img](https://img-blog.csdnimg.cn/img_convert/d29ff175f00f8f0b2342db1c717459da.png#pic_center)

   ​	安装完成后，添加到环境变量PATH中，我的安装路径是：`C:\Program Files\Tesseract-OCR`，

   命令行窗口输入：`tesseract` ，查看是否安装成功。

   ​	2.2 安装Pytesseract

   ```
   pip install pytesseract Pillow
   ```

3. 使用

   ```python
   import pytesseract
   from PIL import Image
   
   result = pytesseract.image_to_string(Image.open(BytesIO(res)), lang='chi_sim')
   print(result)
   ```

4. 参数

   image_to_string：识别图片并将结果转为字符串

   image: 图像对象，推荐使用 `Image`

   lang(String): 语言代码字符串，可通过`pytesseract.get_languages()`查看当前支持的语言

   config(String): 识别图片配置（更多配置自行查找），例如：config='--psm 6'

   ​						所有psm：0：定向脚本监测（OSD）
   ​								1： 使用OSD自动分页
   ​								2 ：自动分页，但是不使用OSD或OCR（Optical Character Recognition，光学字符识别）
   ​								3 ：全自动分页，但是没有使用OSD（默认）
   ​								4 ：假设可变大小的一个文本列。
   ​								5 ：假设垂直对齐文本的单个统一块。
   ​								6 ：假设一个统一的文本块。
   ​								7 ：将图像视为单个文本行。
   ​								8 ：将图像视为单个词。
   ​								9 ：将图像视为圆中的单个词。
   ​								10 ：将图像视为单个字符。

   ​	output_type: 指定输出类型。

   如果需要多个语言包组合并且视为统一的文本块将使用如下参数：

   ```python
   pytesseract.image_to_string(image,lang="chi_sim+eng",config="-psm 6")
   ```

5. 注意

   tesseract 语言包：https://github.com/tesseract-ocr/tessdata

### 4. CnOCR

1. 介绍

   [CnOCR](https://github.com/breezedeus/cnocr)是 Python 3 下的文字识别（Optical Character Recognition，简称OCR）工具包，支持简体中文、繁体中文（部分模型）、英文和数字的常见字符识别，支持竖排文字的识别。自带了20+个训练好的识别模型，适用于不同应用场景，安装后即可直接使用。同时，CnOCR也提供简单的训练命令供使用者训练自己的模型。

   CnOCR 主要针对的是排版简单的印刷体文字图片，如截图图片，扫描件等。目前内置的文字检测和分行模块无法处理复杂的文字排版定位。如果要用于场景文字图片的识别，需要结合其他的场景文字检测引擎使用，例如文字检测引擎 cnstd 。

   **CnOCR** 从 **V2.2** 开始，内部自动调用文字检测引擎 **[CnSTD](https://github.com/breezedeus/cnstd)** 进行文字检测和定位。所以 **CnOCR** V2.2 不仅能识别排版简单的印刷体文字图片，如截图图片，扫描件等，也能识别**一般图片中的场景文字**。

2. 安装

   ```
   pip install cnocr
   ```

3. 使用

   ```python
   from cnocr import CnOcr
   
   img_fp = './huochepiao.jpeg'
   ocr = CnOcr()
   out = ocr.ocr(img_fp)
   print(','.join([i['text'] for i in out]))
   ```

4. 注意

   首次使用cnocr时，系统会自动从[Dropbox](https://link.zhihu.com/?target=https%3A//www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip)下载zip格式的模型压缩文件，win存于 `C:\Users\用户名\AppData\Roaming\cnocr`目录， 下载后的zip文件代码会自动对其解压。

### 5. CnSTD

1. 介绍

   [CnSTD](https://github.com/breezedeus/cnstd)是 Python 3 下的场景文字检测（Scene Text Detection，简称STD）工具包，支持中文、英文等语言的文字检测，自带了多个训练好的检测模型，安装后即可直接使用。

   如需要识别文本框中的文字，可以结合 **OCR** 工具包 **[cnocr](https://github.com/breezedeus/cnocr)** 一起使用。

2. 安装

   ```
   pip install cnstd
   ```

3. 使用

   ```python
   from cnstd import CnStd
   
   std = CnStd()
   img_fp = 'examples/taobao.jpg'
   img = Image.open(img_fp)
   box_info_list = std.detect(img)  # 这里的结果只是
   ```

4. 与CnOCR结合使用

   识别检测框中的文字（OCR），上面示例识别结果中"cropped_img"对应的值可以直接交由 cnocr 中的 **`CnOcr`** 进行文字识别

   ```python
   from cnstd import CnStd
   from cnocr import CnOcr
   
   std = CnStd()
   cn_ocr = CnOcr()
   
   box_infos = std.detect('examples/taobao.jpg')
   
   for box_info in box_infos['detected_texts']:
       cropped_img = box_info['cropped_img']
       ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
       print('ocr result: %s' % str(ocr_res))
   ```

5. 注意

   CnSTD只是检测文本，如果需要获取检测的文本，可通过CnOCR获得。

   首次使用 CnSTD时，系统会自动下载zip格式的模型压缩文件，并存放于 `~/.cnstd`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\cnstd`）。下载速度超快。下载后的zip文件代码会自动对其解压

## 测试

测试图片1：

<img src="https://point.95516.com/quanyiinfo/files/img/2697b4f1c6e8cd38f3b821cbb98245f9.jpg" alt="img" style="zoom: 25%;" />

结果：

很好 》 较好 》 良好 》一般 》 极差

|   测试项    |  效率   | 识别准确度 | 完整度 |                             评价                             |
| :---------: | :-----: | :--------: | :----: | :----------------------------------------------------------: |
|  PaddleOCR  | 6.249s  |    较好    |  80%   |             识别精度不错，但是还有一些没有检测出             |
|   EasyOCR   | 10.681s |    良好    |  95%   |      对场景內文字识别太差，识别结果含有繁体字，速度太慢      |
| PyTesseract | 3.519s  |    较好    |  95%   | 对场景內文字识别太差，但是识别检测完整，识别的结果按照原图片的排版摆放，含有空格与换行符 |
|    CnOCR    | 0.346s  |    极差    |  40%   |      很大一部分没有检测出，识别的准确度也差，但是速度快      |

测试图片2：

<img src="https://point.95516.com/quanyiinfo/files/img/db4c2275ea020be44b30713641719196.jpg" alt="img" style="zoom:67%;" />

结果：

|   测试项    |  效率   | 识别准确度 | 完整度 |                             评价                             |
| :---------: | :-----: | :--------: | :----: | :----------------------------------------------------------: |
|  PaddleOCR  | 9.611s  |    很好    |  99%   |                    准确度很好，与原文相识                    |
|   EasyOCR   | 10.661s |    较好    |  98%   |         准确度也不错，但是个别参杂一些符号，速度太慢         |
| PyTesseract | 1.104s  |    良好    |  85%   | 部分没有识别出，。识别的结果按照原图片的排版摆放，含有空格与换行符 |
|    CnOCR    | 0.496s  |    很好    |  98%   |     准确度很高，除了有一些小的部分没有检测出，并且速度快     |

测试图片3：![img](https://point.95516.com/quanyiinfo/files/img/57c0629686da730da6ed697646b77cb9.jpg)

结果：

|   测试项    |  效率  | 识别准确度 | 完整度 |                            评价                            |
| :---------: | :----: | :--------: | :----: | :--------------------------------------------------------: |
|  PaddleOCR  | 6.854s |    良好    |  95%   |              准确度很好，个别的小字没有识别出              |
|   EasyOCR   | 4.628s |    较好    |  95%   | 准确度也不错，但是个别参杂一些符号，也有一些小字没有识别出 |
| PyTesseract | 0.484s |    极差    |  20%   |      很大部分没有识别出，中文不准确，数字也没有识别出      |
|    CnOCR    | 0.420s |    很好    |  96%   |    准确度很高，除了有一些小的部分没有检测出，并且速度快    |

