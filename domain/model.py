from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from datetime import datetime

import logging
import pandas as pd
import numpy as np
import os


logger = logging.getLogger('model')
logger.setLevel(logging.INFO)

# 로그 포맷
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 로그 출력 스트림 설정
file_handler = logging.FileHandler(filename=f'log_debug_{datetime.now().strftime("%Y-%m-%d")}.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def set_label(elapsed_time=None, pivot=None, is_correct=None):

    label = 0
    if elapsed_time == None and pivot == None and is_correct == None:
        label = 3
    elif elapsed_time <= pivot and is_correct:
        label = 1
    elif elapsed_time > pivot and is_correct:
        label = 2
    elif elapsed_time <= pivot and not is_correct:
        label = 4
    elif elapsed_time > pivot and not is_correct:
        label = 5

    return label


async def set_labels_for_one(df, user_items, q=20):

    '''
    레벨 테스트 결과에 대한 라벨링
    :param df: 기존 사용자들의 풀이 기록 데이터 프레임
    :param user_id: 새로 회원가입한 회원의 아이디
    :param user_items: 새로 회원가입한 회원의 레벨 테스트 문제 풀이 결과
    :return:
    '''

    # userLvtag, userIndex 컬럼 받지 않음
    # user_items = user_items[['USER_ID', 'ELAPSED_TIME', 'PRB_ID', 'USER_LEVEL', 'PRB_USER_ANSW', 'TAG', 'PRB_CORRT_ANSW']]

    questions = df.groupby(df.PRB_ID)

    result_list = []

    for index, row in user_items.iterrows():

        # user_id = row['USER_ID']
        prb_id = row['PRB_ID']

        # NaN 피하기
        if not isinstance(prb_id, str):
            continue

        # 기존에 등장했던 문제의 경우 -> 기존 데이터에 기반하여 라벨링 진행
        if prb_id in questions.groups.keys():

            group = questions.get_group(prb_id)

            # 백분위 20% 구간의 값 찾기
            pivot = np.percentile(group['ELAPSED_TIME'], q)

            label = 0
            elapsed_time = row['ELAPSED_TIME']
            prb_user_answ = row['PRB_USER_ANSW']
            prb_corrt_answ = row['PRB_CORRT_ANSW']

            if elapsed_time <= pivot and prb_user_answ == prb_corrt_answ:
                label = int(1)
            elif elapsed_time > pivot and prb_user_answ == prb_corrt_answ:
                label = int(2)
            elif elapsed_time <= pivot and prb_user_answ != prb_corrt_answ:
                label = int(4)
            elif elapsed_time > pivot and prb_user_answ != prb_corrt_answ:
                label = int(5)

        # 새롭게 등장한 문제의 경우 -> 라벨링을 수행하기 위해 비교할 수 있는 기존의 데이터 없음
        # -> 3으로 라벨링
        else:
            label = int(3)

        user_items.loc[index, 'label'] = label

        # result_list.append(his)

    # 데이터프레임 합치기
    # df = df.append(pd.DataFrame(result_list,
    #                             columns=['USER_ID', 'ELAPSED_TIME', 'PRB_ID', 'USER_LEVEL', 'PRB_USER_ANSW', 'TAG', 'PRB_CORRT_ANSW', 'label'],
    #                             ignore_index=True))
    # return pd.concat([df, pd.DataFrame(result_list)], ignore_index=True)

    # 안 합칠 경우
    # return pd.concat(result_list, ignore_index=True, axis=1)
    return user_items

async def set_labels(df, q=20):

    print('set_labels() ============ >')

    # 문제 ID 별 분류
    questions = df.groupby(df.PRB_ID)
    questions.size()

    result_list = []

    # 그룹별 정/오답 그룹 정규 분포화
    for key, group in questions:

        group = questions.get_group(key)

        # 백분위 20% 구간의 값 찾기
        pivot = np.percentile(group['ELAPSED_TIME'], q)

        group.loc[(group['ELAPSED_TIME'] <= pivot) & (group['PRB_USER_ANSW'] == group['PRB_CORRT_ANSW']), 'label'] = int(1)
        group.loc[(group['ELAPSED_TIME'] > pivot) & (group['PRB_USER_ANSW'] == group['PRB_CORRT_ANSW']), 'label'] = int(2)
        group.loc[(group['ELAPSED_TIME'] <= pivot) & (group['PRB_USER_ANSW'] != group['PRB_CORRT_ANSW']), 'label'] = int(4)
        group.loc[(group['ELAPSED_TIME'] > pivot) & (group['PRB_USER_ANSW'] != group['PRB_CORRT_ANSW']), 'label'] = int(5)

        result_list.append(group)

    return pd.concat(result_list, ignore_index=True)

def learn_model(data_sparse, factor=150, regularization=0.03, iterations=15):

    # data = pd.DataFrame(data, columns=col_names)
    # logger.info(data)
    #
    # # 유저 수
    # data['USER_ID'].nunique()
    # logger.debug("유저 수: {}", data['USER_ID'].nunique())
    #
    # # 문제 수
    # data['PRB_ID'].nunique()
    # logger.debug("문제 수: {}", data['PRB_ID'].nunique())
    #
    # # 데이터 인덱싱
    # user_unique = data['USER_ID'].unique()
    # quest_unique = data['PRB_ID'].unique()
    #
    # # # 유저, 문제 indexing 하는 코드. idx는 index의 약자
    # user_to_idx = {v: k for k, v in enumerate(user_unique)}
    # quest_to_idx = {v: k for k, v in enumerate(quest_unique)}
    #
    # # 인덱싱이 잘 되었는지 확인해 봅니다.
    # # print(user_to_idx['eg93QctMN9ScQ7aJo040afqcor12'])  # 4명의 유저 중 처음 추가된 유저이니 0이 나와야 합니다.
    # # print(quest_to_idx['LV1PQ0041059'])
    #
    # # # user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다.
    # # # 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다.
    # temp_user_data = data['USER_ID'].map(user_to_idx.get).dropna()
    # if len(temp_user_data) == len(data):  # 모든 row가 정상적으로 인덱싱되었다면
    #     logger.info('user_id column indexing OK!!')
    #     data['USER_ID'] = temp_user_data  # data['user_id']을 인덱싱된 Series로 교체해 줍니다.
    # else:
    #     logger.info('user_id column indexing Fail!!')
    #
    # # artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다.
    # temp_artist_data = data['PRB_ID'].map(quest_to_idx.get).dropna()
    # if len(temp_artist_data) == len(data):
    #     logger.info('artist column indexing OK!!')
    #     data['PRB_ID'] = temp_artist_data
    # else:
    #     logger.info('artist column indexing Fail!!')
    #
    # num_user = data['USER_ID'].nunique()
    # num_artist = data['PRB_ID'].nunique()
    #
    # logger.debug(data.USER_ID)
    # logger.debug(data.PRB_ID)
    #
    # csr_data = csr_matrix((data.label, (data.USER_ID, data.PRB_ID)), shape=(num_user, num_artist))
    # logger.info(f"csr_data ::: {csr_data}")

    # Implicit AlternatingLeastSquares 모델의 선언
    als_model = AlternatingLeastSquares(factors=100,
                                        regularization=0.01,
                                        use_gpu=False,
                                        iterations=15,
                                        dtype=np.float32)

    # als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
    csr_data_transpose = data_sparse.T.tocsr()
    logger.info(f'csr_data_transpose ::: {csr_data_transpose}')

    # 모델 훈련
    als_model.fit(csr_data_transpose)

    # 모델 저장
    current_path = os.getcwd()
    if not os.path.exists(f'{current_path}/train'):
        os.mkdir(f'{current_path}/train')
    als_model.save('./train/als-model.npz')

    print(f'als_model.alpha ======== > {als_model.alpha}')
    print(f'als_model.iterations ======= > {als_model.iterations}')
    print(f'als_model.regularization ======= > {als_model.regularization}')
    print(f'als_model.factors ====== > {als_model.factors}')

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

    # return csr_data_transpose, user_to_idx, quest_to_idx
    return csr_data_transpose

def learn_model2(df, factor, regularization, iterations):

    # USER_ID, PRB_ID, LABEL 컬러만 추출하여 새로운 데이터프레임 생성
    data = df[['USER_ID', 'PRB_ID', 'label']]

    # data = pd.DataFrame(data, columns=col_names)
    logger.info(data)

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
    als_model = AlternatingLeastSquares(factors=100,
                                        regularization=0.01,
                                        use_gpu=False,
                                        iterations=15,
                                        dtype=np.float32)

    # als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
    csr_data_transpose = csr_data.T.tocsr()
    logger.info(f'csr_data_transpose ::: {csr_data_transpose}')

    # 모델 훈련
    als_model.fit(csr_data_transpose)

    # 모델 저장
    current_path = os.getcwd()
    if not os.path.exists(f'{current_path}/train'):
        os.mkdir(f'{current_path}/train')
    als_model.save('./train/als-model.npz')

    print(f'als_model.alpha ======== > {als_model.alpha}')
    print(f'als_model.iterations ======= > {als_model.iterations}')
    print(f'als_model.regularization ======= > {als_model.regularization}')
    print(f'als_model.factors ====== > {als_model.factors}')

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

    return als_model

def load_model():

    als_model = AlternatingLeastSquares().load('./train/als-model.npz')

def best_model(df):
    # USER_ID, PRB_ID, LABEL 컬러만 추출하여 새로운 데이터프레임 생성
    data = df[['USER_ID', 'PRB_ID', 'label']]

    # data = pd.DataFrame(data, columns=col_names)
    logger.info(data)

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

    # from implicit.evaluation import leave_k_out_split, precision_at_k, train_test_split

    # Implicit AlternatingLeastSquares 모델의 선언
    als_model = AlternatingLeastSquares(factors=100,
                                        regularization=0.01,
                                        use_gpu=False,
                                        iterations=15,
                                        dtype=np.float32)

    # als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
    csr_data_transpose = csr_data.T.tocsr()
    logger.info(f'csr_data_transpose ::: {csr_data_transpose}')

    # 모델 훈련
    als_model.fit(csr_data_transpose)

    print(f'als_model.alpha ======== > {als_model.alpha}')
    print(f'als_model.iterations ======= > {als_model.iterations}')
    print(f'als_model.regularization ======= > {als_model.regularization}')
    print(f'als_model.factors ====== > {als_model.factors}')

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

    return csr_data_transpose, user_to_idx, quest_to_idx
