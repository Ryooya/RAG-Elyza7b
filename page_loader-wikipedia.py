import re
import requests
import wikipedia
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract


wikipedia.set_lang("ja") #言語を日本語に指定
filename = 'wiki_textfile.txt'

user_input = input("検索したい単語を入力してください。：")
		
_search=wikipedia.search(user_input)
print(_search)

while(True):
    print("表示された一覧から検索したい記事を指定してください(0から数えてね)")
    selected_number = int(input(">>"))
    page = wikipedia.page(_search[selected_number], auto_suggest=False)
    print(page.url)
    print("="*20)
    break

# URLからテキストを取得
document = fetch_url(page.url)
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