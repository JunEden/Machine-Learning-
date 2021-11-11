!pip install opencc-python-reimplemented

from opencc import OpenCC
cc = OpenCC('s2tw')
to_convert =  """繁体中文。
为什么会觉得烦呢？也许让我们学写繁体中文的话，我们可能会烦。
主要看所处的环境。
如果楼主从一出生就学的是繁体中文的话，现在让你学写简体中文，你也会烦的。如果一个英语为母语的国家，让他们的人民学中文，不管是繁体还是简体都会烦。
要你用你的纯母语来书写，应该是最顺手的吧"""
converted = cc.convert(to_convert)
converted

article = """新北市三峽恩主公醫院昨天沒稀釋BNT疫苗原液就對25名民眾施打，遭到懲處停打1周，上午院長吳志雄也與院方幹部出面致歉，表示藥師跟護理師交班不夠仔細，率先施打25瓶疫苗中，因瓶蓋脫落誤認為已經稀釋，才會把6人份的量只給1人施打。

吳志雄解釋，當時藥師在交班時，只說這25瓶要先打，並沒提醒同仁要稀釋，而這25瓶是散裝在袋子，在碰撞下疑似瓶蓋脫落，護理同仁誤以為稀釋過就直接替民眾施打。

25人年齡層曝光：分布在18到65歲 20人在40歲以上
吳志雄指出，這25人年齡介在18到65歲，18到20歲區間有5人，剩下都在40歲以上，年紀最長則是65歲，當中有1人有心臟病史。不過25人中有5人不願到醫院檢查，11人檢查後回家，9人檢查後住院。衛生局表示，這25人皆透過1922疫苗預約平台預約接種BNT疫苗。"""

import jieba
from urllib.request import urlretrieve
url = "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big"
urlretrieve(url, "dict.txt.big")
# 載入大辭典
jieba.set_dictionary("dict.txt.big")
# 載入自定義辭典
jieba.load_userdict("extra.txt")
" ".join(jieba.cut(article))

import jieba.analyse
jieba.analyse.extract_tags(article, 
                           topK=None, 
                           withWeight=True)
 #                          allowPOS=["n"])