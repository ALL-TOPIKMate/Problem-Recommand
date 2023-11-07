# This is a sample Python script.
import os

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from google.cloud.firestore_v1 import FieldFilter
from pytz import timezone, utc # 시간대 설정

import numpy as np
import pandas as pd

KST = timezone('Asia/Seoul')

import logging
import pandas as pd
import json

from dotenv import load_dotenv

load_dotenv() # 환경 변수 로드

# implicit 라이브러리에서 권장하고 있는 부분
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'

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
    st = datetime.now()  # 시간대를 설정하고 현재 시각 불러오기
    current_time = st.strftime('%Y년 %m월 %d일 %H:%M:%S')

    print(f'Daily Recommendation Starts ========== {current_time}')
    logger.info(f'Daily Recommendation Starts ========== {current_time}')

    await train_model()
    await recs_for_all()

    ed = datetime.now()  # 시간대를 설정하고 현재 시각 불러오기
    current_time = ed.strftime('%Y년 %m월 %d일 %H:%M:%S')

    print(f'Daily Recommendation Ends ========== {current_time}')
    logger.info(f'Daily Recommendation Ends ========== {current_time}')


# scheduler = BackgroundScheduler(timezone='Asia/Seoul')
scheduler = AsyncIOScheduler(timezone='Asia/Seoul')
# scheduler.add_job(hello_every_minute, 'cron', second='0')
scheduler.add_job(daily_recs, 'cron', hour=0, minute=0)

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

    # for doc in docs:
    #     print(f'{doc.id} => {doc.to_dict()}')

    # print(f'type(docs) ::: {type(docs)}')
    # logger.info(f'type(docs) ::: {type(docs)}')

    history_dict = list(map(lambda x: x.to_dict(), docs))

    logger.info(f'히스토리 컬렉션 ::: {pd.DataFrame(history_dict)}')
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
        raise Exception(f'존재하지 않는 회원입니다. USER_ID : {user_id}')

    # 레벨테스트 컬렉션 읽어오기
    docs_ref = user_ref.collection('leveltest')
    docs = list(docs_ref.stream())

    # 히스토리 컬렉션에 레벨테스트 기록 추가
    his_col = db.collection('historys')
    logger.info(f'히스토리 컬렉션으로 레벨테스트 풀이 기록을 옮깁니다. =========== >')
    for doc in docs:
        update_time, doc_ref = his_col.add(doc.to_dict())
        logger.info(f'[{update_time}] {doc_ref.id} => {(doc_ref.path)}')

    leveltest_dict = list(map(lambda x: x.to_dict(), docs))

    # print(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(leveltest_dict)}')
    logger.info(f'[{user_id}] 레벨테스트 이력 ::: {pd.DataFrame(leveltest_dict)}')

    return pd.DataFrame(leveltest_dict)

###############################################
########### 전역 변수 설정. app에 등록 ############

app.df = pd.DataFrame() # 매일 자정 수집되는 전체 문제 풀이 이력

app.user_lookup = pd.DataFrame()
app.quest_lookup = pd.DataFrame()

app.data_sparse = pd.DataFrame()
app.data_sparse_trans = pd.DataFrame()

app.users = []
app.questions = []
app.labels = []

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

    logger.info('train_model() start!!!')

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    from domain.model import set_labels, learn_model

    # 데이터 가져오기
    df = await read_document()  # Firebase history 컬렉션에서 풀이 기록 읽어오기

    # 데이터 전처리
    # 1. 풀이 시간에 따라 라벨링을 수행합니다.
    df = await set_labels(df)  # 풀이 시간에 따른 라벨링 수행
    print(f'데이터 라벨링 완료 ::: {df}')

    # 2. 학습에 필요한 컬럼만 추출합니다. 뒤에서 ELAPSED_TIME은 제거할 예정.
    df = df[['USER_ID', 'PRB_ID', 'label', 'ELAPSED_TIME']]

    # 3. 값이 없는 행은 드롭합니다.
    df = df.dropna()

    # 4. 사용자/문제 ID를 숫자 ID로 변환합니다.
    df['USER_IDX'] = df['USER_ID'].astype('category').cat.codes
    df['PRB_IDX'] = df['PRB_ID'].astype('category').cat.codes

    # 5. 숫자 ID로 사용자/문제를 찾을 수 있도록 lookup 데이터를 만듭니다.
    # 사용자 ID lookup
    user_lookup = df[['USER_IDX', 'USER_ID']].drop_duplicates()
    user_lookup['USER_IDX'] = user_lookup.USER_IDX.astype(str)
    app.user_lookup = user_lookup

    # 문제 ID lookup
    quest_lookup = df[['PRB_IDX', 'PRB_ID']].drop_duplicates()
    quest_lookup['PRB_IDX'] = quest_lookup.PRB_IDX.astype(str)
    app.quest_lookup = quest_lookup

    # USER_ID, PRB_ID, label, USER_IDX, PRB_IDX, ELAPSED_TIME 정보를 저장해둡니다.
    app.df = df

    # 학습에 필요한 컬럼만 추출합니다2.
    # df = df[['USER_ID', 'PRB_ID', 'label']]
    df = df.drop(['USER_ID', 'PRB_ID', 'ELAPSED_TIME'], axis=1)
    print(f'Numeric ID로 치환 완료 ::: {df}')

    # 6. 0으로 라벨링된 데이터가 있다면 드롭하고 나머지 데이터만 취합니다.
    df = df.loc[df.label != 0]

    # 7. 전체 사용자, 문제, 라벨 데이터 리스트를 생성합니다.
    app.users = list(np.sort(df['USER_IDX'].unique()))
    app.questions = list(np.sort(df['PRB_IDX'].unique()))
    app.labels = list(df['label'])

    # 사용자-아이템 행렬을 만들기 위한 행, 열 데이터를 얻습니다.
    rows = df['USER_IDX'].astype(int)
    cols = df['PRB_IDX'].astype(int)

    from scipy.sparse import csr_matrix
    app.data_sparse = csr_matrix((app.labels, (rows, cols)), shape=(len(app.users), len(app.questions)))

    # app.csr_data_transpose, app.user_to_idx, app.quest_to_idx = learn_model(app.df)
    app.data_sparse_trans, result = learn_model(app.data_sparse)

    # 사용자 ID 찾기(int -> string)
    # print(f'app.user_lookup.USER_ID.loc[app.user_lookup.USER_IDX == str(0)].iloc[0] ::: {app.user_lookup.USER_ID.loc[app.user_lookup.USER_IDX == str(0)].iloc[0]}')
    # 사용자 IDX 찾기(string -> int)
    # print(f'app.user_lookup.USER_IDX.loc[app.user_lookup.USER_ID == "1aNUrkdm96dQsampFyc9rARsGsR2"].iloc[0] ::: {app.user_lookup.USER_IDX.loc[app.user_lookup.USER_ID == "1aNUrkdm96dQsampFyc9rARsGsR2"].iloc[0]}')

    # app.idx_to_user = {v: k for k, v in app.user_to_idx.items()}
    # app.idx_to_quest = {v: k for k, v in app.quest_to_idx.items()}

    # als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

    logger.info(f'lean_model() result ::: {result}')

    logger.info('train_model() end...')

    return JSONResponse(content=jsonable_encoder(result))

# 전체 유저에게 추천
@app.get('/recs-list')
async def recs_for_all():

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    import numpy as np

    # 저장된 모델을 로드합니다.
    als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')
    logger.info('model load completed !!!')

    # idx_to_user 딕셔너리 -->> np array로 형변환
    # user_idxs = np.fromiter(app.idx_to_user.keys(), dtype=int)
    # questions = app.quest_to_idx.keys()
    # questions = app.quest_lookup.PRB_ID
    # lv1_quest_idx = []
    # lv2_quest_idx = []
    #
    # # 레벨1 문제와 레벨2 문제들을 분리합니다.
    # for question in questions:
    #
    #     quest_idx = app.quest_lookup.PRB_IDX.loc[app.quest_lookup.PRB_ID == question].iloc[0]
    #
    #     if question[:3] == 'LV1':
    #         lv1_quest_idx.append(quest_idx)
    #     else:
    #         lv2_quest_idx.append(quest_idx)

    recs = {}

    # 유저 한 명씩 전체 추천
    # for user_idx in app.users:
    #     # user_id = app.idx_to_user[user_idx]
    #     user_id = app.user_lookup.USER_ID.loc[app.user_lookup.USER_IDX == str(user_idx)].iloc[0] # 사용자 ID
    #
    #     # 유저 탈퇴 여부 확인
    #     user_ref = db.collection('users').document(user_id)
    #     user_doc = user_ref.get()
    #
    #     if not user_doc.exists:
    #         continue
    #
    #     # 유저 레벨 확인
    #     user_level = user_doc.get('my_level') # 사용자 선택 레벨
    #
    #     # 유저 레벨의 문제(아이템) 설정
    #     # items = lv1_quest_idx if user_level == 1 else lv2_quest_idx
    #
    #     # 풀었던 문제 제외 : filter_already_liked_items = True
    #     # 아이템 서브 셋: items = items
    #     '''
    #     itmes: Array of extra item ids. When set this will only rank the items in this array instead of ranking every item the model was fit for. This parameter cannot be used with filter_items
    #     => 추가 항목 ID의 배열입니다. 설정하면 모델에 적합한 모든 항목의 순위를 매기지 않고 이 배열의 항목의 순위만 매깁니다. 이 매개 변수는 filter_items와 함께 사용할 수 없습니다
    #     '''
    #     # ids, scores = als_model.recommend(user_idx,
    #     #                                   app.csr_data_transpose[user_idx],
    #     #                                   filter_already_liked_items=True,
    #     #                                   items=items)
    #     ids, scores = als_model.recommend(user_idx,
    #                                       app.data_sparse_trans[user_idx],
    #                                       filter_already_liked_items=True)

    ids, scores = als_model.recommend(app.users,
                                      app.data_sparse_trans[app.users],
                                      filter_already_liked_items=True)

    print(f'ids ::: {ids}')
    print(f'scores ::: {scores}')

    for i in range(len(ids)):

        recs_list = list()

        user_id = app.user_lookup.USER_ID.loc[app.user_lookup.USER_IDX == str(i)].iloc[0]

        print(f'{user_id} ============= >')
        for j in range(len(ids[i])):

            prb_id = app.quest_lookup.PRB_ID.loc[app.quest_lookup.PRB_IDX == str(ids[i][j])].iloc[0] # 문제 ID

            print(f'PRB_ID: {prb_id}, score: {scores[i][j]}')
            recs_list.append({
                # 'PRB_ID': app.idx_to_quest[ids[i]],
                'PRB_ID': prb_id,
                'SCORE': float(scores[i][j])
            })
        print()

        recs[user_id] = recs_list

        # recs_list = list()
        # print(f'{user_id} ============= >')
        # for i in range(len(ids)):
        #
        #     prb_id = app.quest_lookup.PRB_ID.loc[app.quest_lookup.PRB_IDX == str(ids[i])].iloc[0] # 문제 ID
        #
        #     print(f'PRB_ID: {prb_id}, score: {scores[i]}')
        #     recs_list.append({
        #         # 'PRB_ID': app.idx_to_quest[ids[i]],
        #         'PRB_ID': prb_id,
        #         'SCORE': float(scores[i])
        #     })
        # print()
        #
        # recs[user_id] = recs_list

    logger.info('Recs ===================> {}', recs)

    # Firebase에 저장
    '''
    Firebase 저장 로직.
    각 유저의 rec 컬렉션을 업데이트.
    '''

    for user_id in recs.keys():

        # 유저 탈퇴 여부 확인
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            continue

        '''
        기존 추천 문제 삭제
        '''
        # 컬렉션 참조
        rec_collection = db.collection('users').document(user_id).collection('recommend')

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

        # 추천 문제 풀이 상태 초기화. userCorrect: 0, userIndex: 0
        rec_collection.document('Recommend').set({'userCorrect': 0, 'userIndex': 0})

    return JSONResponse(content=jsonable_encoder(recs))

# 단일 유저에게 추천
@app.post('/recs-list/{user_id}')
async def recs_for_one(user_id):

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    from domain.model import set_labels_for_one, learn_model
    from implicit.als import AlternatingLeastSquares
    from implicit.recommender_base import RecommenderBase

    als_model = AlternatingLeastSquares(RecommenderBase).load('./train/als-model.npz')

    # 데이터 가져오기
    df = await read_temp_document(user_id)  # Firebase leveltest 컬렉션에서 풀이 기록 읽어오기

    # 데이터 전처리
    # 1. 풀이 시간에 따른 라벨링을 수행합니다.
    df = await set_labels_for_one(app.df, df)

    # 2. 학습에 필요한 컬럼만 추출합니다.
    df = df[['USER_ID', 'PRB_ID', 'label']]

    # 3. 값이 없는 행은 드롭합니다.
    df = df.dropna()

    # 4. 사용자/문제 ID를 숫자 ID로 변환합니다.
    df['USER_IDX'] = df['USER_ID'].astype('category').cat.codes
    df['PRB_IDX'] = df['PRB_ID'].astype('category').cat.codes

    # 5. 숫자 ID로 사용자/문제를 찾을 수 있도록 lookup 데이터를 만듭니다.
    # 사용자 ID lookup
    user_lookup = df[['USER_IDX', 'USER_ID']].drop_duplicates()
    user_lookup['USER_IDX'] = user_lookup.USER_IDX.astype(str)

    # 문제 ID lookup
    quest_lookup = df[['PRB_IDX', 'PRB_ID']].drop_duplicates()
    quest_lookup['PRB_IDX'] = quest_lookup.PRB_IDX.astype(str)

    df = df.drop(['USER_ID', 'PRB_ID'], axis=1)
    print(f'Numeric ID로 치환 완료 ::: {df}')

    # 6. 0으로 라벨링된 데이터가 있다면 드롭하고 나머지 데이터만 취합니다.
    df = df.loc[df.label != 0]

    # 7. 전체 사용자, 문제, 라벨 데이터 리스트를 생성합니다.
    users = list(np.sort(df['USER_IDX'].unique()))
    questions = list(np.sort(df['PRB_IDX'].unique()))
    labels = list(df['label'])

    # 사용자-아이템 행렬을 만들기 위한 행, 열 데이터를 얻습니다.
    rows = df['USER_IDX'].astype(int)
    cols = df['PRB_IDX'].astype(int)

    from scipy.sparse import csr_matrix
    data_sparse = csr_matrix((labels, (rows, cols)), shape=(len(users), len(questions)))
    # 전치해야 하나??
    # data_sparse_trans = data_sparse.T.tocsr()

    # lv1_quest_idx = []
    # lv2_quest_idx = []
    #
    # # 레벨1 문제와 레벨2 문제들을 분리합니다.
    # for question in questions:
    #
    #     quest_idx = app.quest_lookup.PRB_IDX.loc[app.quest_lookup.PRB_ID == question].iloc[0]
    #
    #     if question[:3] == 'LV1':
    #         lv1_quest_idx.append(quest_idx)
    #     else:
    #         lv2_quest_idx.append(quest_idx)

    # 유저 탈퇴 여부 확인
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise Exception(f'존재하지 않는 회원입니다. USER_ID : {user_id}')

    # 유저 레벨 확인
    user_level = user_doc.get('my_level')  # 사용자 선택 레벨

    # 유저 레벨의 문제(아이템) 설정
    # items = lv1_quest_idx if user_level == 1 else lv2_quest_idx

    # ids, scores = als_model.recommend(userid=0,
    #                                   user_items=app.data_sparse_trans[0],
    #                                   N=10,
    #                                   recalculate_user=True,
    #                                   filter_already_liked_items=True)

    # 개별 추천
    # 해당 유저의 데이터만 사용.
    # 풀었던 문제 제외 : filter_already_liked_items = True
    # 아이템 서브 셋: items = items
    '''
    itmes: Array of extra item ids. When set this will only rank the items in this array instead of ranking every item the model was fit for. This parameter cannot be used with filter_items
    => 추가 항목 ID의 배열입니다. 설정하면 모델에 적합한 모든 항목의 순위를 매기지 않고 이 배열의 항목의 순위만 매깁니다. 이 매개 변수는 filter_items와 함께 사용할 수 없습니다
    '''
    ids, scores = als_model.recommend(userid=0,
                                      user_items=data_sparse[0],
                                      N=10,
                                      recalculate_user=True,
                                      filter_already_liked_items=True)

    recs = {}
    recs_list = list()
    print(f'{user_id} ============= >')
    for i in range(len(ids)):
        prb_id = app.quest_lookup.PRB_ID.loc[app.quest_lookup.PRB_IDX == str(ids[i])].iloc[0]  # 문제 ID

        print(f'PRB_ID: {prb_id}, score: {scores[i]}')
        recs_list.append({
            # 'PRB_ID': app.idx_to_quest[ids[i]],
            'PRB_ID': prb_id,
            'SCORE': float(scores[i])
        })
    print()

    recs[user_id] = recs_list

    logger.info('Recs ===================> {}', recs)


    # 추천 문제 recommend 컬렉션에 저장
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

    # 추천 문제 풀이 상태 초기화. userCorrect: 0, userIndex: 0
    rec_collection.document('Recommend').set({'userCorrect': 0, 'userIndex': 0})

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

