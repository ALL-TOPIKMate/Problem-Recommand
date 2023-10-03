# This is a sample Python script.
from fastapi import FastAPI

# FastAPI app 생성
app = FastAPI()

from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 로그 포맷
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 로그 출력 스트림 설정
file_handler = logging.FileHandler(filename=f'log_debug_{datetime.now().strftime("%Y-%m-%d")}.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Firebase 클라이언트 생성
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

import pandas as pd
import numpy as np
import os

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

# Firebase CRUD 구현

def create_document():

    doc_ref = db.collection('historys').document('test_id')
    doc_ref.set({
        u'USER_ID': u'test_id',
        u'ELAPSED_TIME': 8000,
        u'PRB_CORRT_ANSW': u'3',
        u'PRB_ID': u'LV1PQ0000010',
        u'TAG': u'002',
        u'PRB_USER_ANSW': 1
    })


def read_document():

    docs_ref = db.collection(u'historys')
    docs = list(docs_ref.stream())

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')

    # print(f'type(docs) ::: {type(docs)}')
    # logger.info(f'type(docs) ::: {type(docs)}')

    history_dict = list(map(lambda x: x.to_dict(), docs))

    # print(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(history_dict)}')
    # logger.info(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(history_dict)}')
    return pd.DataFrame(history_dict)


'''
매번 라벨링(풀이 시간을 이진화)하는 작업이 필요할 것 같은데..

모범 답안을 기준으로
우리만의 평가 기준으로 가는 게 맞을 듯

할 수 있다는 것을 보여주는 것이 중요할 듯
'''


def update_document():

    doc_ref = db.collection('historys').document('test_id')
    doc_ref.update({
        u'PRB_USER_ANSW': 2
    })


def delete_document():

    doc_ref = db.collection('historys').document('test_id')
    doc_ref.delete()

## 데이터 전처리 - 라벨 만들기
def set_labels(df):

    # solved = read_document()

    # logger.info(f'solved ::: {solved}')

    # 문제 ID 별 분류
    questions = df.groupby(df.PRB_ID)
    questions.size()

    result_list = []

    # 그룹별 정/오답 그룹 정규 분포화
    idx = 0
    for key, group in questions:

        # if (idx == 1):
        #     break;

        # print(f'[{key}] ============= ')

        # group = questions.get_group(key)

        # 정답 그룹
        correct_group = group[group['PRB_USER_ANSW'] == group['PRB_CORRT_ANSW']]
        # 오답 그룹
        wrong_group = group[group['PRB_USER_ANSW'] != group['PRB_CORRT_ANSW']]

        # print(f'group.shape: {group.shape}, correct_group.shape: {correct_group.shape}, wrong_group.shape: {wrong_group.shape}')

        pivot = 0
        # 정답 그룹 pivot 값
        if correct_group.size > 0:

            # N% 구간의 기준값 찾기
            pivot = np.percentile(correct_group['ELAPSED_TIME'], 15)
            try:
                correct_group.loc[correct_group['ELAPSED_TIME'] <= pivot, 'label'] = 1
                correct_group.loc[correct_group['ELAPSED_TIME'] > pivot, 'label'] = 2
            except ValueError as e:
                logger.error('error: ', e)

            # result_df = pd.concat([result_df, correct_group], axis=0)
            result_list.append(correct_group)

        # print('correct_group: ', correct_group)

        # labeled1 = correct_group
        # labeled1['label'] = np.where(labeled1['elapsed_time'] <= pivot, 1, 2)

        pivot2 = 0
        # 오답 그룹 pivot 값
        if wrong_group.size > 0:
            pivot2 = np.percentile(wrong_group['ELAPSED_TIME'], 15)
            try:
                wrong_group.loc[wrong_group['ELAPSED_TIME'] <= pivot2, 'label'] = 4
                wrong_group.loc[wrong_group['ELAPSED_TIME'] > pivot2, 'label'] = 5
            except ValueError as e:
                logger.error('error: ', e)
            # result_df = pd.concat([result_df, wrong_group], axis=0)
            result_list.append(wrong_group)

        # print('wrong_group: ', wrong_group)
        # labeled2 = wrong_group
        # labeled2['label'] = np.where(labeled2['elapsed_time'] <= pivot2, 5, 4)

        # merge
        # temp = pd.concat([labeled1, labeled2], axis=0)
        # temp = pd.concat([correct_group, wrong_group], axis=0)

        idx += 1

    return pd.concat(result_list, ignore_index=True)

user_to_idx = dict()
quest_to_idx = dict()

csr_data_transpose = pd.DataFrame()

def learn_model(df):
    data = df[['USER_ID', 'PRB_ID', 'label']]

    # data = pd.DataFrame(data, columns=col_names)
    logger.info(data)

    # USER_ID, PRB_ID, LABEL 컬러만 추출하여 새로운 데이터프레임 생성
    # data = df[['USER_ID', 'PRB_ID', 'label']]
    # logger.info(f'필요한 컬럼만 추출한 상태 ::: {data}')
    # logger.info(f'data ::: {data}')

    # 유저 수
    data['USER_ID'].nunique()
    logger.debug("유저 수: {}", data['USER_ID'].nunique())

    # 문제 수
    data['PRB_ID'].nunique()
    logger.debug("문제 수: {}", data['PRB_ID'].nunique())

    # 데이터 인덱싱
    user_unique = data['USER_ID'].unique()
    quest_unique = data['PRB_ID'].unique()

    # # 유저, 문제 indexing 하는 코드. idx는 index의 약자
    user_to_idx = {v: k for k, v in enumerate(user_unique)}
    quest_to_idx = {v: k for k, v in enumerate(quest_unique)}

    # 인덱싱이 잘 되었는지 확인해 봅니다.
    # print(user_to_idx['eg93QctMN9ScQ7aJo040afqcor12'])  # 4명의 유저 중 처음 추가된 유저이니 0이 나와야 합니다.
    # print(quest_to_idx['LV1PQ0041059'])

    # # user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다.
    # # 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다.
    temp_user_data = data['USER_ID'].map(user_to_idx.get).dropna()
    if len(temp_user_data) == len(data):  # 모든 row가 정상적으로 인덱싱되었다면
        logger.info('user_id column indexing OK!!')
        data['USER_ID'] = temp_user_data  # data['user_id']을 인덱싱된 Series로 교체해 줍니다.
    else:
        logger.info('user_id column indexing Fail!!')

    # artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다.
    temp_artist_data = data['PRB_ID'].map(quest_to_idx.get).dropna()
    if len(temp_artist_data) == len(data):
        logger.info('artist column indexing OK!!')
        data['PRB_ID'] = temp_artist_data
    else:
        logger.info('artist column indexing Fail!!')

    num_user = data['USER_ID'].nunique()
    num_artist = data['PRB_ID'].nunique()

    logger.debug(data.USER_ID)
    logger.debug(data.PRB_ID)

    csr_data = csr_matrix((data.label, (data.USER_ID, data.PRB_ID)), shape=(num_user, num_artist))
    logger.info(f"csr_data ::: {csr_data}")

    # implicit 라이브러리에서 권장하고 있는 부분
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Implicit AlternatingLeastSquares 모델의 선언
    als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False, iterations=15,
                                        dtype=np.float32)

    # als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
    csr_data_transpose = csr_data.T.tocsr()
    logger.info(f'csr_data_transpose ::: {csr_data_transpose}')

    # 모델 훈련
    als_model.fit(csr_data_transpose)
    als_model.save('./train/als-model.npz')

    # logger.info('========= 모델 훈련 끝 =========')
    #
    # # 비슷한 문제 찾기
    # favorite_artist = 'LV1PQ0041015'
    # artist_id = quest_to_idx[favorite_artist]
    # print('quest_to_idx ::: ', quest_to_idx)
    # # print('artist_id ::: ', artist_id)
    # similar_artist = als_model.similar_items(artist_id, N=3)
    # print('similar_artist ::: ', similar_artist)
    #
    # idx_to_artist = {v: k for k, v in quest_to_idx.items()}
    # temp = [idx_to_artist[i] for i in similar_artist[0]]
    # print(temp)

    # user = user_to_idx['eg93QctMN9ScQ7aJo040afqcor12']
    # recommend에서는 user*item CSR Matrix를 받습니다.
    # print(als_model.recommend_all(user, 3))
    # print('csr_data_transpose[user] ::: ', csr_data_transpose[user])
    # artist_recommended = als_model.recommend(user, csr_data_transpose[user], N=3, filter_already_liked_items=True)
    # print('artist_recommended ::: ', artist_recommended)

    # index to artist
    # print(*[idx_to_artist[i] for i in artist_recommended[0]])
    #
    # # 추천 기여도 확인
    # rihanna = artist_to_idx['LV2PQ0052003']
    # explain = als_model.explain(user, csr_data, itemid=rihanna)
    #
    # [(idx_to_artist[i[0]], i[1]) for i in explain[1]]

    # temp = als_model.recommend_all(csr_data_transpose(data['USER_ID'].unique()), 10)
    # return temp

def load_model():

    return AlternatingLeastSquares().load('./train/als-model.npz')

# 잡 스케줄링
from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler

def hello_every_minute():
    now = datetime.now()
    current_time = now.strftime('%Y년 %m월 %d일 %H:%M:%S')

    print(f'hello now is {current_time}')
    logger.info(f'hello now is {current_time}')

@app.on_event('startup')
def init_app():

    # 스케줄링 테스트 - apscheduler 라이브러리
    scheduler = BlockingScheduler(timezone='Asia/Seoul')
    scheduler.add_job(hello_every_minute, 'cron', second='0')
    scheduler.start()

@app.get("/")
def home():

    return 'Hello! This is TOPIK Mate recommendation server!'

@app.get("/model")
def test():

    # df = read_document()  # Firebase history 컬렉션에서 풀이 기록 읽어오기
    # df = set_labels(df)  # 풀이 시간에 따른 라벨링 수행
    #
    # learn_model(df)

    # create_document()
    doc_ref = db.collection('historys').document('test_id')
    doc_ref.set({
        u'USER_ID': u'test_id',
        u'ELAPSED_TIME': 8000,
        u'PRB_CORRT_ANSW': u'3',
        u'PRB_ID': u'LV1PQ0000010',
        u'TAG': u'002',
        u'PRB_USER_ANSW': 1
    })

    return 'Successfully created model!'

@app.post("/daily-recs/{user_id}")
def daily_recs(user_id):

    als_model = load_model()

    user = user_to_idx[user_id]
    # recommend에서는 user*item CSR Matrix를 받습니다.
    print(als_model.recommend_all(user, 3))
    print('csr_data_transpose[user] ::: ', csr_data_transpose[user])
    artist_recommended = als_model.recommend(user, csr_data_transpose[user], N=3, filter_already_liked_items=True)
    print('artist_recommended ::: ', artist_recommended)


#########################################################
# 스케줄링 테스트 - sched 라이브러리

# scheduler = sched.scheduler(time.time, time.sleep)
# scheduler.enter(0, 1, hello_every_minute, ())
#
# scheduler.run()

# 스케줄링 테스트 - apscheduler 라이브러리
# scheduler.start()

#################### 이 밑으로는 실행 안됨
