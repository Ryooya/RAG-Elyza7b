# RAGありローカルLLMを作ろう

表題の通りやってきましょう。
# もくじ
- [RAGありローカルLLMを作ろう](#ragありローカルllmを作ろう)
- [もくじ](#もくじ)
- [実行環境](#実行環境)
- [パッケージ](#パッケージ)
        - [注意点](#注意点)
- [コードの説明](#コードの説明)
    - [1. モデルとファイルの設定・ドキュメント読み込み、分割](#1-モデルとファイルの設定ドキュメント読み込み分割)
    - [2. 埋め込みとベクターストアの初期化](#2-埋め込みとベクターストアの初期化)
    - [3. モデルのロードと設定](#3-モデルのロードと設定)
    - [4. プロンプトテンプレートの定義](#4-プロンプトテンプレートの定義)
    - [5. パイプラインの設定](#5-パイプラインの設定)
    - [6.  質問応答チェーンの作成](#6--質問応答チェーンの作成)
    - [7. 実行部](#7-実行部)
- [実際にやってみた](#実際にやってみた)
    - [余談](#余談)
        - [初音ミクのwikipediaを学習](#初音ミクのwikipediaを学習)
- [まとめ](#まとめ)


# 実行環境
- win11
- PC
    - 13th Gen Intel core i9-13900K
    - RTX 3090 Ti
- Python 3.12.3
- VScode
- conda 24.5.0

# パッケージ
あとで`requirement.txt`用意します。  
pytorchはご自分の環境に合わせてインストールしてください。  
##### 注意点
`faiss-gpu`のみ、公式でcondaまたはCmakeでビルドが推奨されています。エラーが出たときはanacondaから環境構築するとよいです。
```{パッケージ類}
langchain
langchain-community
transformers
huggingface-hub
trafilatura
accelerate
bitsandbytes
pypdf
tiktoken
sentence_transformers
faiss-gpu
```
実行してみてないので足りないものがあったら小林まで教えて下さい。

# コードの説明

`page_loader.py`で、指定されたURLからテキストを取得し、テキストを文字のみの形に処理し、`textfile.txt`として出力します。  
ここからは`main.py`のコードを分割して解説していきます。
### 1. モデルとファイルの設定・ドキュメント読み込み、分割
以下に示すコードは、使用するファイル名、モデルの保存ディレクトリ、モデルのID、埋め込みモデルのIDの設定をまず行います。  
その次に、`TextLoader` を使用して `textfile.txt` を読み込み、 `CharacterTextSplitter` を使用してテキストをチャンクに分割しています。  
ここでは、`text_splitter`に設定した通り、改行で分割し、チャンクサイズは300文字、重なりは20文字となるようにしています。


```python
filename = 'textfile.txt'
MODEL_SAVE_DIR = "model_save"
model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
emb_model_id = "intfloat/multilingual-e5-large"
loader = TextLoader(filename, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=300,
    chunk_overlap=20,
)
texts = text_splitter.split_documents(documents)
print(f"Number of text chunks: {len(texts)}")
```
separatorを改行ではなく句読点にしたらまた違った結果になりそうかな。

### 2. 埋め込みとベクターストアの初期化
HuggingFaceの埋め込みモデルを使用してテキストチャンクを埋め込みに変換し、FAISSを使用してベクターストアを作成します。  
ベクターストアから類似したチャンクを検索するための下準備として、レトリーバーを設定します。ここでは最も類似する3つのチャンクを取得します。

```python
embeddings = HuggingFaceEmbeddings(model_name=emb_model_id)
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})
```

### 3. モデルのロードと設定
まあ、4bit量子化とモデルのロードと設定ですね。ついでにCUDAの確認もしてます。
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    cache_dir=MODEL_SAVE_DIR
).eval()

if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("CUDA is not available, using CPU")
```


### 4. プロンプトテンプレートの定義
だんだんだれてきました。ここはElyzaのUsageに則って書いてます。ほかのモデルを試したかったらここを変える必要がありますね。
```python
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。"
text = "{context}\n=======\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
```
### 5. パイプラインの設定
langchainを使う上で必須の項目。ここは公式のドキュメントを丸々コピーしたのであんまり意味は分かってないです。

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
)
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string"
)
```

### 6.  質問応答チェーンの作成
RAGを動かす部分。`RetrievalQA`は設定が豊富なのでいろいろ試したい。
```python
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(
        pipeline=pipe,
        # model_kwargs=dict(temperature=0.1, do_sample=True, repetition_penalty=1.1)
    ),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
)
```
### 7. 実行部
関数化してmainにするならここ。  
ユーザーからの質問を受け取り、モデルに入力して回答を生成する。生成した回答とともに、質問応答チェーンを使用してRAG (Retrieval-Augmented Generation)を行い、回答とソースドキュメントを表示するようにしてあります。  
RAG無しRAG有りも比較できます。
```python
while True:
    print("文章を入力してください")
    input_text = input(">>")
    start = time.time()
    inputs = template.format(context='', question=input_text)
    inputs = tokenizer(inputs, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
    print('RAG無し回答:', output)
    result = qa(input_text)
    print('RAG有り回答:', result['result'])
    print('='*10)
    print('ソース:', result['source_documents'])
    print('='*10)
    print("経過時間:", time.time() - start)
```


# 実際にやってみた
とりあえず、ワンピースのwikipediaを読み込ませてみる。
```
import re
from trafilatura import fetch_url, extract

#url = "https://w.wiki/AEwG" # 長岡高専のWikipedia
url = "https://ja.wikipedia.org/wiki/ONE_PIECE" #ONEPIECEのWikipedia

filename = 'textfile.txt'

# URLからテキストを取得
document = fetch_url(url)
text = extract(document)

# テキストの前処理
def clean_text(text):

    # 特定の不要な文字や記号を取り除く
    text = re.sub(r'\[.*?\]', '', text)  # 角括弧で囲まれたテキスト（例えば[1]のような参照）を削除
    text = re.sub(r'\(.*?\)', '', text)  # 括弧で囲まれたテキストを削除
    text = re.sub(r'[^a-zA-Z0-9ぁ-んァ-ン一-龥。、・！？\n\s]', '', text)  # 日本語や英数字、句読点、改行以外の文字を削除

    # 追加のクリーニング処理が必要な場合はここに追加
    # 例: URLやメールアドレスを削除
    text = re.sub(r'http\S+', '', text)  # URLを削除
    text = re.sub(r'\S+@\S+', '', text)  # メールアドレスを削除

    return text

cleaned_text = clean_text(text)
print(cleaned_text[:1000])

# クリーンアップされたテキストを保存
with open(filename, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)
```
これを実行してみると～
```
ONE PIECE

ジャンル  少年漫画・海賊・冒険ファンタジ・バトル


漫画

作者  尾田栄一郎

出版社  集英社


掲載誌  週刊少年ジャンプ

レベル  ジャンプ・コミックス

発表号  1997年34号

発表期間  1997年7月22日

巻数  既刊108巻2024年3月4日現在

テンプレト  ノト
プロジェクト  漫画

ポタル  漫画

ONE PIECEワンピスは、尾田栄一郎による日本の少年漫画作品。週刊少年ジャンプ集英社にて1997年34号から連載中。略称はワンピ。
概要
海賊王を夢見る少年モンキ・D・ルフィを主人公とする、ひとつなぎの大秘宝ワンピスを巡る海洋冒 険ロマン。
夢への冒険・仲間たちとの友情といったテマを前面に掲げ、バトルやギャグシン、感動エピソドをメインとする少年漫画の王道を行く物語として人気を博している。また、長年にわたりながら深く練り込まれた壮大な世界観・巧緻な設定のストリも特徴。
2024年3月の時点で単行本は第108巻まで刊行されており、週刊少年ジャンプ歴代作品の中ではこちら葛飾区亀有公園前派出所1976年  2016年に次ぐ長期連載となっている。国内累計発行部数は2022年時点で日本の漫画では最高となる4億1000万部を突破している。また、第67巻は初版発行部数405万部の国内出版史上最高記録を樹立し、第57巻2010年3月発売以降の単行本は初版300万部以上発行を継続するなど、出版の国内最高記録をいくつも保持している。
2015年6月15日には Most Copies Published For The Same Comic Book Series By A Single Author 最も多く発行された単一作者によるコミックシリズ 名義でギネス世界記録に認定された。実績は発 行部数3億2086万6000部2014年12月時点。なお、このギネス世界記録は2022年7月付で同作品によって更新され、日本では同年8月に日本国内累計発行部数4億1656万6000部と報道された。
本作とともに長年ジャンプの看板作品であったNARUTO ナルト同様、海外での人気も高い。海外では 翻訳版が60以上の国と地域で販売されており、海外でのコミックス累計発行部数は2022年8月時点で1億部を
```

`print(cleaned_text[:1000])`で一応中身を出力してみてます。  
いい感じですね。  
それでは、このテキストデータを使って実行してみましょう。

実行した結果を以下に示します。
```
Number of text chunks: 428
Loading checkpoint shards: 100%|███████████████████████████| 2/2 [00:04<00:00,  2.23s/it] 
Using CUDA
文章を入力してください
>>ニコ・ロビンの職業は？
RAG無し回答: [INST] <<SYS>>
参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。
<</SYS>>


=======
ユーザからの質問は次のとおりです。ニコ・ロビンの職業は？ [/INST]  ニコ・ロビンの職業は、冒険者です。
C:\anaconda\envs\LLM\lib\site-packages\langchain_core\_api\deprecation.py:119: LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 0.3.0. Use invoke instead.
  warn_deprecated(


> Entering new RetrievalQA chain...

> Finished chain.
RAG有り回答: <s>[INST] <<SYS>>
参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。
<</SYS>>

空島編
24巻  32巻
ジャヤ編24巻  25巻
アラバスタを後にしたルフィ達は、B・W社副社長であった考古学者ニコ・ロビンを仲間に加える。次の島に向かう航海中、突如空から巨大なガレオン船が落下し、記録指針ログポスの指す進路が上向きに変更される。それは伝説とされる空に浮かぶ島空島への指針を意味していた。

トニトニ・チョッパ
声  大谷育江
麦わらの一味船医。ヒトヒトの実を食べ人の能力を持った人間トナカイ。万能薬何でも治せる医者を目指している。
ニコ・ロビン
声  山口由里子
麦わらの一味考古学者。ハナハナの実の能力者。空白の100年の謎を解き明かすため旅をしている。 
フランキ
声  矢尾一樹
麦わらの一味船大工。体中に武器を仕込んだサイボグ。自分の作った船に乗り、その船が海の果てに辿り着くのを見届けることが夢。

THE 6TH LOG ARABASTA2006年3月発行、ISBN 4081110255
THE 7TH LOG VIVI2006年4月発行、ISBN 4081110263
THE 8TH LOG SKYPIEA2008年4月発行、ISBN 9784081110278
THE 9TH LOG GOD2008年5月発行、ISBN 9784081110285
THE 10TH LOG BELL2008年6月発行、ISBN 9784081110292
THE 11TH LOG WATER SEVEN2009年4月発行、ISBN 9784081110094
THE 12TH LOG ROCKET MAN2009年5月発行、ISBN 9784081110100
THE 13TH LOG NICO ROBIN2009年7月発行、ISBN 9784081110117
THE 14TH LOG FRANKY2009年8月発行、ISBN 9784081110124
THE 15TH LOG THRILLER BARK2011年2月発行、ISBN 9784081110339
THE 16TH LOG BROOK2011年2月発行、ISBN 9784081110353
=======
ユーザからの質問は次のとおりです。ニコ・ロビンの職業は？ [/INST]  質問に回答いたします。  

ニコ・ロビンの職業は「考古学者」です。
==========
ソース: [Document(page_content='空島編\n24巻  32巻\nジャヤ編24巻  25巻\nアラバスタを後にしたルフィ達は、B・W社副社長であった考古学者ニコ・ロビンを仲間に加える。次の島に向かう航海中、突如空から巨大なガレオン船が落下し、記録指針ログポスの指す進路が上向きに変更される。それは伝説とされる空に浮かぶ島空島への指針を意味していた。', metadata={'source': 'textfile.txt'}), Document(page_content='トニトニ・チョッパ\n声  大谷育江\n麦わらの一味船医。ヒトヒトの実を食べ人の能力を持った人間トナカイ。万能薬何でも治せる医者を目指している。\nニコ・ロビン\n声  山口由里子\n麦わらの一味考古学者。ハナハナの実の能力者。空白の100年の謎を解き明かす ため旅をしている。\nフランキ\n声  矢尾一樹\n麦わらの一味船大工。体中に武器を仕込んだサイボグ。自分の作った船に乗り、その船が海の果てに辿り着くのを見届けることが夢。', metadata={'source': 'textfile.txt'}), Document(page_content='THE 6TH LOG ARABASTA2006年3月発行、ISBN 4081110255\nTHE 7TH LOG VIVI2006年4月発行、ISBN 4081110263\nTHE 8TH LOG SKYPIEA2008年4月発行、ISBN 9784081110278\nTHE 9TH LOG GOD2008年5月発行、ISBN 9784081110285\nTHE 10TH LOG BELL2008年6月発行、ISBN 9784081110292\nTHE 11TH LOG WATER SEVEN2009年4月発行、ISBN 9784081110094\nTHE 12TH LOG ROCKET MAN2009年5月発行、ISBN 9784081110100\nTHE 13TH LOG NICO ROBIN2009年7月発行、ISBN 9784081110117\nTHE 14TH LOG FRANKY2009年8月発行、ISBN 9784081110124\nTHE 15TH LOG THRILLER BARK2011年2月発行、ISBN 9784081110339\nTHE 16TH LOG BROOK2011年2月発行、ISBN 9784081110353', metadata={'source': 'textfile.txt'})]
==========
経過時間: 16.38978624343872
```

結論として、導き出した回答はそれぞれRAGあり、なしで"冒険者"、"考古学者"でしたね。正解は”考古学者”なので、  
これはRAGを用いてハルシネーションを回避した、ということが言えそうです。

さて、それでは自分の好きなサイトのURLから情報を得てRAG機能の有効性を確かめてみましょう。

### 余談
ここからは実際に遊んでみたものの結果一覧です。
##### 初音ミクのwikipediaを学習
| Q                                         | RAGなし                                                                                                                                                                                                                               | RAGあり                                                                                                                                                                                                                                                                                            | 
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | 
| 初音ミクは何歳ですか                      | 初音ミクは、2007年7月2日に誕生したため、2021年2月現在で14歳です。                                                                                                                                                                     |  承知しました。初音ミクは16歳です。<br><br>2024年1月22日のクリプトンのブログ記事によると、16歳のミクのアニバーサリーライブ「MIKU FES24春Happy 16th Birthday」が331日後の日本武道館で開催される予定であるとのことです。                                                                             | 
| BUMPOFCHICKENとコラボしたのはいつですか？ | BUMP OF CHICKENとコラボしたのは2017年11月22日に発売されたアルバム「THE LAST ACOUSTIC」に収録されている「終わりなき物語」です。                                                                                                        | 2014年の7月31日に東京ドムで開催されたBUMP OF CHICKENのコンサトツアファイナルWILLPOLIS 2014にゲストとして登場し、コラボ楽曲rayをBUMP OF CHICKENと共に披露した。                                                                                                                                     | 
| 初音ミクとネギの関係を教えてください      |  初音ミクとネギの関係について回答いたします 。<br><br>初音ミクは、VOCALOIDのキャラクターであり、ネギは、「とある魔術の禁書目録」に登場するキャラクターです。<br><br>2つのキャラクターは関係ないため、該当する回答をいたしません。<br> | 初音ミクとネギの関係について回答いたします 。<br><br>ネギはミクの定番アイテムとしてファンの間で広く受け入れられているとのことです。これは派生キャラクタのはちゅねミクを生み出し た動画VOCALOID2 初音ミクにIevan Polkkaを歌わせてみたで、ミクに長ネギを持たせていたことから広まったとされています。 | 

# まとめ
- langchainを使えば比較的簡単に実装できた。
- RAGの精度向上に関してはまだ記事とかで明確に示されているものがないので、論文を読むべきかも？