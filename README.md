# ELMo_implementation

## Explanation
[大規模日本語ビジネスニュースコーパスを学習したELMo（MeCab利用）モデルの利用方法と精度比較検証](https://qiita.com/kaeru_nantoka/items/bca53a2daea2b29c9b39)を参考にして、テキスト分類をELMoで行う。  

## Preparation
- [このGoogle Drive](https://drive.google.com/drive/u/1/folders/1sau1I10rFeAn8BDk8eZDL5qaEjTlNghp)から「文字単位・単語単位埋め込みモデル」内のencoder.pklとtoken_embedder.pklをcharフォルダに、「単語単位埋め込みモデル」内のencoder.pklとtoken_embedder.pklをwordフォルダに入れる。
- ライブラリのインストールを行う。
```
pip install -r requirements.txt
```
