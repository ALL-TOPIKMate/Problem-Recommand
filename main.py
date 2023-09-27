# This is a sample Python script.
from fastapi import FastAPI

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# FastAPI app 생성
app = FastAPI()

# Firebase 클라이언트 생성
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

import pandas as pd
import numpy as np

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

    docs_ref = db.collection('historys')
    docs = list(docs_ref.stream())

    # for doc in docs:
    #     print(f'{doc.id} => {doc.to_dict()}')

    logger.info(f'type(docs) ::: {type(docs)}')

    history_dict = list(map(lambda x: x.to_dict(), docs))

    logger.info(f'pd.DataFrame(history_dict) ::: {pd.DataFrame(history_dict)}')

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


# 잡 스케줄링
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 로그 포맷
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 로그 출력 스트림 설정
file_handler = logging.FileHandler(filename=f'log_debug_{datetime.now().strftime("%Y-%m-%d")}.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

scheduler = BlockingScheduler(timezone='Asia/Seoul')

@scheduler.scheduled_job('cron', second='0', id='schedule_test')
def hello_every_minute():
    now = datetime.now()
    current_time = now.strftime('%Y년 %m월 %d일 %H:%M:%S')

    print(f'hello now is {current_time}')
    logger.info(f'hello now is {current_time}')

# ALS 학습

## 데이터 전처리 - 라벨 만들기
def set_labels():

    solved = read_document()

    logger.info(f'solved ::: {solved}')

    # 문제 ID 별 분류
    questions = solved.groupby(solved.PRB_ID)
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
                print('error: ', e)

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
                print('error: ', e)
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    read_document()

    # df = set_labels() # 라벨이 설정된 상태의 데이터프레임
    # print(f'라벨 설정한 상태 ::: {df}')

    # # USER_ID, PRB_ID, LABEL 컬러만 추출하여 새로운 데이터프레임 생성
    # data = df[['USER_ID', 'PRB_ID', 'label']]
    # print(f'필요한 컬럼만 추출한 상태 ::: {data}')

    # train_data(data)

    # 비슷한 문제 찾기
    # favorite_artist = 'LV2PQ0052050'
    # artist_id = artist_to_idx[favorite_artist]
    # similar_artist = als_model.similar_items(artist_id, N=3)
    # print('similar_artist ::: ', similar_artist)
    #
    # idx_to_artist = {v: k for k, v in artist_to_idx.items()}
    # [idx_to_artist[i[0]] for i in similar_artist]
    #
    # user = user_to_idx['eg93QctMN9ScQ7aJo040afqcor12']
    # # recommend에서는 user*item CSR Matrix를 받습니다.
    # artist_recommended = als_model.recommend(user, csr_data, N=3, filter_already_liked_items=True)
    # print('artist_recommended ::: ', artist_recommended)

    # index to artist
    # print(*[idx_to_artist[i[0]] for i in artist_recommended])

    # # 추천 기여도 확인
    # rihanna = artist_to_idx['LV2PQ0052003']
    # explain = als_model.explain(user, csr_data, itemid=rihanna)
    #
    # [(idx_to_artist[i[0]], i[1]) for i in explain[1]]

    # https://ceuity.tistory.com/31
    # https://kmhana.tistory.com/31

    # 스케줄링 테스트 - sched 라이브러리

    # scheduler = sched.scheduler(time.time, time.sleep)
    # scheduler.enter(0, 1, hello_every_minute, ())
    #
    # scheduler.run()

    # 스케줄링 테스트 - apscheduler 라이브러리
    scheduler.start()

    #################### 이 밑으로는 실행 안됨

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
