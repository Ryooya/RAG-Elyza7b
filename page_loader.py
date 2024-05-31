import re
from trafilatura import fetch_url, extract

url = "https://w.wiki/AEwG" # 長岡高専のWikipedia
filename = 'textfile.txt'

# URLからテキストを取得
document = fetch_url(url)
text = extract(document)

# テキストの前処理
def clean_text(text):
    # 改行や余分な空白を取り除く
    #text = re.sub(r'\n+', '\n', text)  # 複数の改行を1つの改行にまとめる
    #text = re.sub(r'\s+', ' ', text)  # 複数の空白を1つの空白にまとめる

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
