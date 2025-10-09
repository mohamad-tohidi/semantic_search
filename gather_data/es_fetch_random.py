import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()


ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_URL = os.getenv("ES_URL")



query = {
    "query": {
        "match_all": {}
    },
    "size": 10,
    "_source": {"excludes": ["*_vector"]},
}

es = Elasticsearch(
    hosts=[{"host": ES_URL, "port": 9200, "scheme": "https"}],
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)

INDEX_NAME = "parsaqa_questions_003"

response = es.search(
    index=INDEX_NAME,
    body=query,
)

print(response)


# qa_questions_003', '_id': '9809', '_score': 1.0, '_source': {'metadata': {'viewer': 0, 'was_not_question': False, 'title': 'فردى
#  بل از ازدواج دايم، ازدواج موقّت کرده است و توافقى که روى مهريّه عقد دايم شده بود را سهوا مهريّه عقد موقّت قرار داده\u200cاند، تکليف چ
#  ست؟'}, 'modified_time': '2025-08-21T10:32:24.016496+00:00', 'question': {'metadata': {'augmentations': {'negative': 'فردى قبل از
#  زدواج دايم، ازدواج موقّت کرده است و توافقى که روى مهريّه عقد دايم شده بود را سهوا مهريّه عقد موقّت قرار داده\u200cاند، چه نتیجه\u200c
#  ی خواهد داشت؟', 'positive': 'وقوعاً، فردی قبل از عقد ازدواج دائم، به طور موقت ازدواج نموده و مبلغی که به عنوان مهریه برای ازدواج د
#  ئم توافق شده بود، به اشتباه به عنوان مهریه برای ازدواج موقت تعیین شده است. در این شرایط، تکلیف قانونی این مبلغ و تعهدات طرفین چه
#  واهد بود؟'}, 'category': 'احکام و فقه', 'url': 'https://bahjat.ir/fa/ahkam/7756', 'source_name': 'bahjat', 'source_link': 'bahjat.ir'}, 'language': 'fa', 'text': {'fa': 'فردى قبل از ازدواج دايم، ازدواج موقّت کرده است و توافقى که روى مهريّه عقد دايم شده بود را س
#  وا مهريّه عقد موقّت قرار داده\u200cاند، تکليف چيست؟'}}, 'answers': [[{'metadata': {'augmentations': {}, 'source_name': 'bahjat', 'source_link': 'bahjat.ir'}, 'language': 'fa', 'text': {'fa': 'احتياط در اين است که شوهر مدت عقد موقت را ببخشد و زن هم او را از مهر
#  برا کند و باز عقد موقّت را تجديد کنند.'}}]], 'elastic_id': '9809'}}]}}