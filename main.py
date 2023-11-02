# This is a sample Python script.
import os

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from google.cloud.firestore_v1 import FieldFilter
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

##########################################
########### 잡 스케줄링 ####################

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
# scheduler.add_job(hello_every_minute, 'cron', second='0')
scheduler.add_job(daily_recs, 'cron', hour='0')

##########################################
############ 앱 시작/종료 ##################

@app.on_event('startup')
async def init_app():
    # 스케줄링 테스트 - apscheduler 라이브러리
    scheduler.start()

@app.on_event('shutdown')
async def shutdown_app():
    scheduler.remove_all_jobs()


###############################################
##################### DB 관련 #################
async def read_document():

    '''
    히스토리 컬렉션 read
    :return: 히스토리 컬렉션 안의 문서 데이터 프레임
    '''

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

async def read_temp_document(user_id):
    '''
    레벨테스트 결과 데이터 read
    :param user_id: 사용자 ID
    :return: 결과 데이터 DataFrame
    '''

    # 탈퇴 여부 확인
    user_ref = db.collection(u'users').document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise Exception

    # 레벨테스트 컬렉션 읽어오기
    docs_ref = user_ref.collection('leveltest')
    docs = list(docs_ref.stream())

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')

    print(f'type(docs) ::: {type(docs)}')
    # logger.info(f'type(docs) ::: {type(docs)}')

    leveltest_dict = list(map(lambda x: x.to_dict(), docs))

    print(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(leveltest_dict)}')
    # logger.info(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(history_dict)}')

    return pd.DataFrame(leveltest_dict)

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

from evaluation.evaluation import leave_k_out_split

@app.get('/')
async def home():

    return {'message': 'Hello! Server is healthy!'}

@app.post('/model')
async def train_model():

    from domain.model import set_labels, learn_model

    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    # 데이터 가져오기
    df = await read_document()  # Firebase history 컬렉션에서 풀이 기록 읽어오기
    app.df = await set_labels(df)  # 풀이 시간에 따른 라벨링 수행

    print(f'app.df ::: {app.df}')
    app.csr_data_transpose, app.user_to_idx, app.quest_to_idx = learn_model(app.df)


    print('app.csr_data_transpose ::: ', app.csr_data_transpose)
    print('type(csr_data_transpose) ::: ', type(app.csr_data_transpose))

    app.idx_to_user = {v: k for k, v in app.user_to_idx.items()}
    app.idx_to_quest = {v: k for k, v in app.quest_to_idx.items()}

    # als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

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
    user_idxs = np.fromiter(app.idx_to_user.keys(), dtype=int)
    questions = app.quest_to_idx.keys()
    lv1_quest_idx = []
    lv2_quest_idx = []

    for question in questions:
        if question[:3] == 'LV1':
            lv1_quest_idx.append(app.quest_to_idx[question])
        else:
            lv2_quest_idx.append(app.quest_to_idx[question])

    recs = {}

    # 유저 한 명씩 전체 추천
    for user_idx in user_idxs:
        user_id = app.idx_to_user[user_idx]

        # 유저 탈퇴 여부 확인
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            continue

        # 유저 레벨 확인
        user_level = user_doc.get('my_level')

        # 유저 레벨의 문제(아이템) 설정
        items = lv1_quest_idx if user_level == '1' else lv2_quest_idx

        print(f'items ::: {items}')
        print(f'als_model.item_factors ::: {als_model.item_factors}')
        print(f'len(als_model.item_factors) ::: {len(als_model.item_factors)}')

        # ids, scores = als_model.recommend(user_idx,
        #                                   app.csr_data_transpose[user_idx],
        #                                   filter_already_liked_items=True,
        #                                   items=items)
        ids, scores = als_model.recommend(user_idx,
                                          app.csr_data_transpose[user_idx],
                                          filter_already_liked_items=True)


        recs_list = list()
        print(f'{app.idx_to_user[user_idx]} ============= >')
        for i in range(len(ids)):
            print(f'PRB_ID: {app.idx_to_quest[ids[i]]}, score: {scores[i]}', end=', ')
            recs_list.append({
                'PRB_ID': app.idx_to_quest[ids[i]],
                'SCORE': float(scores[i])
            })

        recs[app.idx_to_user[user_idx]] = recs_list

    # !recommend_all은 deprecated 됨. 대신 recommend()에서 userids를 np array로 넘겨서 사용하면 됨
    # ids, scores = als_model.recommend(user_idxs, app.csr_data_transpose[user_idxs], 10, filter_already_liked_items=True, items=)
    # recs = {}
    #
    # for i in range(len(ids)):
    #
    #     recs_list = list()
    #     print(f'{app.idx_to_user[i]} ============= >')
    #     for j in range(len(ids[i])):
    #         print(f'PRB_ID: {app.idx_to_quest[j]}, score: {scores[i][j]}', end=', ')
    #         recs_list.append({
    #             'PRB_ID': app.idx_to_quest[j],
    #             'score': float(scores[i][j])
    #         })
    #     print()
    #
    #     recs[app.idx_to_user[i]] = recs_list

    logger.info('Recs ===================> {}', recs)

    # Firebase에 저장
    '''
    Firebase 저장 로직.
    각 유저의 rec 컬렉션을 업데이트.
    '''

    for user_id in recs.keys():

        '''
        기존 추천 문제 삭제
        '''
        # 컬렉션 참조
        rec_collection = db.collection('users').document(user_id).collection('recommend')

        # 테스트로 한명의 유저것만 업데이트
        if user_id == 'WLdH1AfmPMaRsRLJZMdlR8gPJNm2':
            
            # 컬렉션 내의 모든 문서 가져옵니다.
            rec_docs = rec_collection.stream()

            # 기존 추천 문제를 삭제합니다.
            for doc in rec_docs:
                doc.reference.delete()

            # 새로운 오늘의 추천 문제를 저장합니다.
            for question in recs[user_id]:

                prb_id = question['PRB_ID']

                '''
                문제 ID로 문제 데이터 가져오는 로직
                '''
                prb_level = prb_id[:3]
                prb_source1 = prb_id[3:5]
                prb_source2 = prb_id[5:9]

                prb_ref = (db.collection('problems')
                            .document(prb_level)
                            .collection(prb_source1)
                            .document(prb_source2)
                            .collection('PRB_LIST')
                            .document(prb_id))

                prb_doc = prb_ref.get()

                if not prb_doc.exists:
                    raise Exception(f'존재하지 않는 도큐먼트. DOC_PATH = {prb_doc.path}')

                # 문제 데이터를 얻습니다.
                prb_data = prb_doc.to_dict()

                # 문제 데이터를 추천 문제 컬렉션에 그대로 삽입합니다.
                rec_collection.document(prb_id).set(prb_data)

            # 추천 문제 풀이 상태 초기화. userCorrect: 0, userIndex: 10
            rec_collection.document('Recommend').set({'userCorrect': 0, 'userIndex': 10})

    return JSONResponse(content=jsonable_encoder(recs))

@app.post('/recs-list/{user_id}')
async def recs_for_one(user_id):

    from domain.model import set_labels_for_one, learn_model

    # 데이터 가져오기
    df = await read_temp_document(user_id)  # Firebase leveltest 컬렉션에서 풀이 기록 읽어오기
    df = await set_labels_for_one(app.df, user_id, df)  # 풀이 시간에 따른 라벨링 수행

    # 유저 선택 레벨 구하기
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    user_level = user_doc.get('my_level')

    # 유저 레벨에 맞는 문제들만 거르기
    questions = app.quest_to_idx.keys()
    items = []
    for question in questions:
        if question[2] == str(user_level):
            items.append(app.quest_to_idx[question])

    # 유저 인덱스 구하기
    user_idx = len(app.user_to_idx) # 기존 마지막 유저 인덱스 + 1

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    import numpy as np

    als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

    # CSR sparse matrix로 바꿔주기
    from scipy.sparse import coo_matrix, csr_matrix

    # 데이터 인덱싱
    user_unique = df['USER_ID'].unique()
    quest_unique = df['PRB_ID'].unique()

    # 유저, 문제 indexing 하는 코드. idx는 index의 약자
    user_to_idx = {v: k for k, v in enumerate(user_unique)}
    quest_to_idx = {v: k for k, v in enumerate(quest_unique)}

    # 인덱싱이 잘 되었는지 확인해 봅니다.
    # print(user_to_idx['eg93QctMN9ScQ7aJo040afqcor12'])  # 4명의 유저 중 처음 추가된 유저이니 0이 나와야 합니다.
    # print(quest_to_idx['LV1PQ0041059'])

    # # user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다.
    # # 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다.
    temp_user_data = df['USER_ID'].map(user_to_idx.get).dropna()
    if len(temp_user_data) == len(df):  # 모든 row가 정상적으로 인덱싱되었다면
        logger.info('user_id column indexing OK!!')
        df['USER_ID'] = temp_user_data  # data['user_id']을 인덱싱된 Series로 교체해 줍니다.
    else:
        logger.info('user_id column indexing Fail!!')

    # artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다.
    temp_artist_data = df['PRB_ID'].map(quest_to_idx.get).dropna()
    if len(temp_artist_data) == len(df):
        logger.info('artist column indexing OK!!')
        df['PRB_ID'] = temp_artist_data
    else:
        logger.info('artist column indexing Fail!!')

    df = df[['USER_ID', 'PRB_ID', 'label']]
    num_user = df['USER_ID'].nunique()
    num_quest = df['PRB_ID'].nunique()

    csr_data = csr_matrix((df.label, (df.USER_ID, df.PRB_ID)), shape=(num_user, num_quest))
    csr_data_transpose = csr_data.T.tocsr()

    # 최초 추천 문제 생성 - recalculate_user=True
    # !recommend_all은 deprecated 됨. 대신 recommend()에서 userids를 np array로 넘겨서 사용하면 됨
    user_idx = user_to_idx[user_id]
    # ids, scores = als_model.recommend(userid=user_idx,
    #                                   user_items=csr_data_transpose[user_idx],
    #                                   N=10,
    #                                   filter_already_liked_items=True,
    #                                   recalculate_user=True,
    #                                   items=items)
    ids, scores = als_model.recommend(userid=user_idx,
                                      user_items=csr_data_transpose[user_idx],
                                      N=10,
                                      filter_already_liked_items=True,
                                      recalculate_user=True)

    recs = {}
    recs_list = []

    for i, quest_idx in enumerate(ids):

        print(f'PRB_ID: {app.idx_to_quest[quest_idx]}, score: {scores[i]}')
        recs_list.append({
            'PRB_ID': app.idx_to_quest[quest_idx],
            'score': float(scores[i])
        })

    recs[user_id] = recs_list

    logger.info('Recs ===================> {}', recs)

    # Firebase에 저장

    rec_collection = db.collection('users').document(user_id).collection('recommend')

    for question in recs[user_id]:
        prb_id = question['PRB_ID']

        '''
        문제 ID로 문제 데이터 가져오는 로직
        '''
        prb_level = prb_id[:3]
        prb_source1 = prb_id[3:5]
        prb_source2 = prb_id[5:9]

        prb_ref = (db.collection('problems')
                   .document(prb_level)
                   .collection(prb_source1)
                   .document(prb_source2)
                   .collection('PRB_LIST')
                   .document(prb_id))

        prb_doc = prb_ref.get()

        if not prb_doc.exists:
            raise Exception(f'존재하지 않는 도큐먼트. DOC_PATH = {prb_doc.path}')

        # 문제 데이터를 얻습니다.
        prb_data = prb_doc.to_dict()

        # 문제 데이터를 추천 문제 컬렉션에 그대로 삽입합니다.
        rec_collection.document(prb_id).set(prb_data)

    # 추천 문제 풀이 상태 초기화. userCorrect: 0, userIndex: 10
    rec_collection.document('Recommend').set({'userCorrect': 0, 'userIndex': 10})

    return JSONResponse(content=jsonable_encoder(recs))


# 부분 학습.
# 새롭게 회원가입한 회원에 대하여 모델 부분 학습 후, 추천 문제 반환
@app.post('/model/user/{user_id}')
async def partial_fit_for_new_user(user_id):

    from domain.model import set_labels_for_one, learn_model

    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

    # 임시 컬렉션에서 레벨 테스틑 풀이 결과 로드
    docs_ref = db.collection('users').document(user_id).collection('leveltest').where(filter=FieldFilter('id', '!=', 'Leveltest'))

    docs = list(docs_ref.stream())

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')

    print(f'type(docs) ::: {type(docs)}')

    leveltest_dict = list(map(lambda x: x.to_dict(), docs))

    leveltest_df = pd.DataFrame(leveltest_dict)

    # 레벨 테스트 문제 풀이 기록에 라벨링
    user_df = await set_labels_for_one(app.df, user_id, leveltest_df)

    # 사용자 아이디 -> 인덱싱
    data = user_df[['USER_ID', 'PRB_ID', 'label']]
    print(f'data ::: {data}')

    # 임시 인덱스 생성
    user_to_idx_temp = app.user_to_idx
    idx_to_user_temp = app.idx_to_user
    quest_to_idx_temp = app.quest_to_idx
    idx_to_quest_temp = app.idx_to_quest

    # 임시로 현재 사용자에 대한 인덱스 매핑
    idx_to_user_temp[len(user_to_idx_temp)] = user_id # 인덱스: 사용자ID
    user_to_idx_temp[user_id] = len(user_to_idx_temp) # 사용자ID: 인덱스

    user_idx = user_to_idx_temp[user_id] # 사용자 인덱스

    als_model.partial_fit_users(user_idx, user_df)

    return {'message': '부분학습 완료!'}

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
