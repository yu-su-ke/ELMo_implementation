import re


def tokenizer(text, j_t):
    return [tok for tok in j_t.tokenize(text, wakati=True)]


def clean_text(x):
    x = str(x)
    x = x.replace('\n', '')  # 改行削除
    x = x.replace('\t', '')  # タブ削除
    x = x.replace('年', '')
    x = x.replace('月', '')
    x = x.replace('日', '')
    x = x.replace('時', '')
    x = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', x)
    x = re.sub(r'\[math\]', ' LaTex math ', x)  # LaTex削除
    x = re.sub(r'\[\/math\]', ' LaTex math ', x)  # LaTex削除
    x = re.sub(r'\\', ' LaTex ', x)  # LaTex削除
    x = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', x)  # タグの削除
    x = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', x)  # タグの削除
    x = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', x)  # タグの削除
    x = re.sub(' +', ' ', x)  # 連続して出現する空白の削除
    return x


def rm_puncts(text):
    puncts = r',.":)・《》「」『』！(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】〜'
    for punct in puncts:
        text = text.replace(punct, '')
    return text


def rm_spaces(text):
    spaces = ['\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000',
              '\x10', '\x7f', '\x9d', '\xad',
              '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88', '\x8d', '\x80', '\x8e', '\x9a',
              '\x94', '\xa0',
              '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96', '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',
              ]
    for space in spaces:
        text = text.replace(space, ' ')
    return text


def replace_num(text):
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    text = re.sub('[０-９]{5,}', '#####', text)
    text = re.sub('[０-９]{4}', '####', text)
    text = re.sub('[０-９]{3}', '###', text)
    text = re.sub('[０-９]{2}', '##', text)
    return text


def preprocess(df_col, j_t):
    df_col = rm_puncts(df_col)
    df_col = rm_spaces(df_col)
    df_col = clean_text(df_col)
    df_col = replace_num(df_col)
    df_col = tokenizer(df_col, j_t)
    return df_col
