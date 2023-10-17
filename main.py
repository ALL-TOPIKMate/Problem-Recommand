# This is a sample Python script.
import os

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
from pytz import timezone, utc # 시간대 설정

KST = timezone('Asia/Seoul')

import logging
import pandas as pd
import json

from dotenv import load_dotenv

load_dotenv() # 환경 변수 로드

# 로깅 시간대 설정을 위한 컨버터 함수
def timetz(*args):
    return datetime.now(KST).timetuple()

# 로그 포맷 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 컨버터 설정
logging.Formatter.converter = timetz

# 로그 출력 스트림 설정
file_handler = logging.FileHandler(filename=f'log_debug_{datetime.now(KST).strftime("%Y-%m-%d")}.log')
file_handler.setFormatter(formatter)

logger = logging.getLogger('main')
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# Firebase 클라이언트 생성
import firebase_admin
from firebase_admin import credentials, firestore

my_credential = {
  "type": "service_account",
  "project_id": "topik-mate",
  "private_key_id": os.getenv('PRIVATE_KEY_ID'),
  "private_key": os.getenv('PRIVATE_KEY').replace(r'\n', '\n'),
  "client_email": os.getenv('CLIENT_EMAIL'),
  "client_id": os.getenv('CLIENT_ID'),
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": os.getenv('AUTH_PROVIDER_X509_CERT_URL'),
  "client_x509_cert_url": os.getenv('CLIENT_X509_CERT_URL'),
  "universe_domain": "googleapis.com"
}

# cred = credentials.Certificate("./serviceAccountKey.json")
cred = credentials.Certificate(my_credential)
firebase_admin.initialize_app(cred)
db = firestore.client()

# FastAPI app 생성
app = FastAPI()

# 잡 스케줄링

def hello_every_minute():
    now = datetime.now(KST) # 시간대를 설정하고 현재 시각 불러오기
    current_time = now.strftime('%Y년 %m월 %d일 %H:%M:%S')

    print(f'hello now is {current_time}')
    logger.info(f'hello now is {current_time}')

async def daily_recs():
    now = datetime.now(KST)  # 시간대를 설정하고 현재 시각 불러오기
    current_time = now.strftime('%Y년 %m월 %d일 %H:%M:%S')

    print(f'Daily Recommendation ========== {current_time}')
    logger.info(f'Daily Recommendation ========== {current_time}')

    await train_model()
    await recs_for_all()

# scheduler = BackgroundScheduler(timezone='Asia/Seoul')
scheduler = AsyncIOScheduler(timezone='Asia/Seoul')
scheduler.add_job(hello_every_minute, 'cron', second='0')
# scheduler.add_job(daily_recs, 'cron', minute='0')

@app.on_event('startup')
async def init_app():
    # 스케줄링 테스트 - apscheduler 라이브러리
    scheduler.start()

@app.on_event('shutdown')
async def shutdown_app():
    scheduler.remove_all_jobs()


###############################################
async def read_document():

    docs_ref = db.collection(u'historys')
    docs = list(docs_ref.stream())

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')

    print(f'type(docs) ::: {type(docs)}')
    # logger.info(f'type(docs) ::: {type(docs)}')

    history_dict = list(map(lambda x: x.to_dict(), docs))

    print(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(history_dict)}')
    # logger.info(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(history_dict)}')

    return pd.DataFrame(history_dict)

###############################################
########### 전역 변수 설정. app에 등록 ############

app.df = pd.DataFrame() # 매일 자정 수집되는 전체 문제 풀이 이력

app.user_to_idx = {}
app.idx_to_user = {}
app.quest_to_idx = {}
app.idx_to_quest = {}

from scipy.sparse import csr_matrix
# scipy.sparse.csr_matrix()

app.csr_data_transpose = None

###############################################

@app.get('/')
async def home():

    return {'message': 'Hello! Server is healthy!'}

@app.post('/model')
async def train_model():

    from domain.model import set_labels, learn_model

    # 데이터 가져오기
    df = await read_document()  # Firebase history 컬렉션에서 풀이 기록 읽어오기
    app.df = await set_labels(df)  # 풀이 시간에 따른 라벨링 수행

    print(f'app.df ::: {app.df}')
    app.csr_data_transpose, app.user_to_idx, app.quest_to_idx = learn_model(app.df)


    print('app.csr_data_transpose ::: ', app.csr_data_transpose)
    print('type(csr_data_transpose) ::: ', type(app.csr_data_transpose))

    app.idx_to_user = {v: k for k, v in app.user_to_idx.items()}
    app.idx_to_quest = {v: k for k, v in app.quest_to_idx.items()}

    return {'message': '모델 학습 완료!'}

@app.get('/recs-list')
async def recs_for_all():

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    import numpy as np

    als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

    # idx_to_user 딕셔너리 -->> np array로 형변환
    userids = np.fromiter(app.idx_to_user.keys(), dtype=int)

    # !recommend_all은 deprecated 됨. 대신 recommend()에서 userids를 np array로 넘겨서 사용하면 됨
    ids, scores = als_model.recommend(userids, app.csr_data_transpose[userids], 10, filter_already_liked_items=True)
    recs = {}

    for i in range(len(ids)):

        recs_list = list()
        print(f'{app.idx_to_user[i]} ============= >')
        for j in range(len(ids[i])):
            print(f'PRB_ID: {app.idx_to_quest[j]}, score: {scores[i][j]}', end=', ')
            recs_list.append({
                'PRB_ID': app.idx_to_quest[j],
                'score': float(scores[i][j])
            })
        print()

        recs[app.idx_to_user[i]] = recs_list

    logger.info('Recs ===================> {}', recs)

    # Firebase에 저장
    '''
    Firebase 저장 로직.
    각 유저의 rec 컬렉션을 업데이트.
    '''

    for user_id in recs.keys():

        # 하나만 테스트
        if user_id == 'WLdH1AfmPMaRsRLJZMdlR8gPJNm2':
            rec_collection = db.collection('users').document(user_id).collection('recommend')

            for question in recs[user_id]:

                prb_id = question['PRB_ID']
                rec_collection.document(prb_id).set(question)

            # 추천 문제 풀이 상태 초기화. userCorrect: 0, userIndex: 10
            rec_collection.document('Recommend').update({'userCorrect': 0, 'userIndex': 10})

    return JSONResponse(content=jsonable_encoder(recs))

# 부분 학습.
# 새롭게 회원가입한 회원에 대하여 모델 부분 학습 후, 추천 문제 반환
@app.post('/model/user/{user_id}')
async def partial_fit_for_new_user(user_id, user_items):

    from domain.model import set_labels_for_one, learn_model

    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

    user_df = await set_labels_for_one(app.df, user_id, user_items)

    # als_model.recalculate_user(user_id)

    # data.label, (data.USER_ID, data.PRB_ID))

    als_model.partial_fit_users(user_id, user_df)

    pass

# 새롭게 업로드된 문제에 대하여 추천 모델 업데이트
@app.post('/model/question')
def recalculate_for_new_question():

    pass

@app.post('/test')
async def test():

    from domain.model import set_labels, best_model

    # 데이터 가져오기
    df = await read_document()  # Firebase history 컬렉션에서 풀이 기록 읽어오기
    df = await set_labels(df)  # 풀이 시간에 따른 라벨링 수행

    print(f'df ::: {df}')
    best_model(df)


    # print('app.csr_data_transpose ::: ', app.csr_data_transpose)
    # print('type(csr_data_transpose) ::: ', type(app.csr_data_transpose))

    # app.idx_to_user = {v: k for k, v in app.user_to_idx.items()}
    # app.idx_to_quest = {v: k for k, v in app.quest_to_idx.items()}

    return {'message': '테스트 모델 학습 완료!'}

# @app.get("/insert-test")
# async def create_document():
#
#     # doc_ref = db.collection('historys').document('test_id')
#     # doc_ref.set({
#     #     u'USER_ID': u'test_id',
#     #     u'ELAPSED_TIME': 8000,
#     #     u'PRB_CORRT_ANSW': u'3',
#     #     u'PRB_ID': u'LV1PQ0000010',
#     #     u'TAG': u'002',
#     #     u'PRB_USER_ANSW': 1
#     # })
#
#     return {"message": "end!!"}

# @app.get("/model")
# def test():
#
#     # df = read_document()  # Firebase history 컬렉션에서 풀이 기록 읽어오기
#     # df = set_labels(df)  # 풀이 시간에 따른 라벨링 수행
#     #
#     # learn_model(df)
#
#     # create_document()
#     # doc_ref = db.collection('historys').document('test_id')
#     # doc_ref.set({
#     #     u'USER_ID': u'test_id',
#     #     u'ELAPSED_TIME': 8000,
#     #     u'PRB_CORRT_ANSW': u'3',
#     #     u'PRB_ID': u'LV1PQ0000010',
#     #     u'TAG': u'002',
#     #     u'PRB_USER_ANSW': 1
#     # })
#     #
#     return 'Successfully created model!'
#


# @app.post("/daily-recs/{user_id}")
# def daily_recs(user_id):
#
#     als_model = load_model()
#
#     user = user_to_idx[user_id]
#     # recommend에서는 user*item CSR Matrix를 받습니다.
#     print(als_model.recommend_all(user, 3))
#     print('csr_data_transpose[user] ::: ', csr_data_transpose[user])
#     artist_recommended = als_model.recommend(user, csr_data_transpose[user], N=3, filter_already_liked_items=True)
#     print('artist_recommended ::: ', artist_recommended)
