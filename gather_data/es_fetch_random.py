import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from models import QARecord


load_dotenv()


ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_URL = os.getenv("ES_URL")



es = Elasticsearch(
    hosts=[{"host": ES_URL, "port": 9200, "scheme": "https"}],
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)

INDEX_NAME = "parsaqa_questions_003"

NUM_SHORTEST = int(os.getenv("NUM_SHORTEST", "5"))
NUM_LONGEST = int(os.getenv("NUM_LONGEST", "5"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "2000"))
SEARCH_REQUEST_TIMEOUT = int(os.getenv("SEARCH_REQUEST_TIMEOUT", "60"))

source_filter = {"excludes": ["*_vector"]}

random_sample_query = {"match_all": {}}

sampled = es.search(
    index=INDEX_NAME,
    size=SAMPLE_SIZE,
    query=random_sample_query,
    _source=source_filter,
    track_total_hits=False,
    request_timeout=SEARCH_REQUEST_TIMEOUT,
)

def extract_length(doc_source):
    try:
        return len(doc_source["question"]["text"]["fa"])  
    except Exception:
        return 0

hits = sampled.get("hits", {}).get("hits", [])
scored = [(extract_length(h.get("_source", {})), h) for h in hits]
scored.sort(key=lambda x: x[0])

shortest_hits = [h for _, h in scored[:NUM_SHORTEST]]
longest_hits = [h for _, h in scored[-NUM_LONGEST:]] if NUM_LONGEST > 0 else []

seen = set()
combined_hits = []
for h in shortest_hits + longest_hits:
    _id = h.get("_id")
    if _id not in seen:
        seen.add(_id)
        combined_hits.append(h)


data = combined_hits[-1]
qa_record = QARecord.model_validate(data["_source"])

print(qa_record)



# qa_questions_003', '_id': '9809', '_score': 1.0, '_source': {'metadata': {'viewer': 0, 'was_not_question': False, 'title': 'فردى
#  بل از ازدواج دايم، ازدواج موقّت کرده است و توافقى که روى مهريّه عقد دايم شده بود را سهوا مهريّه عقد موقّت قرار داده\u200cاند، تکليف چ
#  ست؟'}, 'modified_time': '2025-08-21T10:32:24.016496+00:00', 'question': {'metadata': {'augmentations': {'negative': 'فردى قبل از
#  زدواج دايم، ازدواج موقّت کرده است و توافقى که روى مهريّه عقد دايم شده بود را سهوا مهريّه عقد موقّت قرار داده\u200cاند، چه نتیجه\u200c
#  ی خواهد داشت؟', 'positive': 'وقوعاً، فردی قبل از عقد ازدواج دائم، به طور موقت ازدواج نموده و مبلغی که به عنوان مهریه برای ازدواج د
#  ئم توافق شده بود، به اشتباه به عنوان مهریه برای ازدواج موقت تعیین شده است. در این شرایط، تکلیف قانونی این مبلغ و تعهدات طرفین چه
#  واهد بود؟'}, 'category': 'احکام و فقه', 'url': 'https://bahjat.ir/fa/ahkam/7756', 'source_name': 'bahjat', 'source_link': 'bahjat.ir'}, 'language': 'fa', 'text': {'fa': 'فردى قبل از ازدواج دايم، ازدواج موقّت کرده است و توافقى که روى مهريّه عقد دايم شده بود را س
#  وا مهريّه عقد موقّت قرار داده\u200cاند، تکليف چيست؟'}}, 'answers': [[{'metadata': {'augmentations': {}, 'source_name': 'bahjat', 'source_link': 'bahjat.ir'}, 'language': 'fa', 'text': {'fa': 'احتياط در اين است که شوهر مدت عقد موقت را ببخشد و زن هم او را از مهر
#  برا کند و باز عقد موقّت را تجديد کنند.'}}]], 'elastic_id': '9809'}}]}}