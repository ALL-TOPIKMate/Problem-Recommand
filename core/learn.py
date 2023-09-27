
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import os

col_names = ['USER_ID', 'PRB_ID', 'LABEL']
data = [('eg93QctMN9ScQ7aJo040afqcor12', 'LV1PQ0041015', 4.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV1PQ0041040', 4.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV1PQ0041049', 4.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV1PQ0041057', 4.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV1PQ0041059', 4.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV1PQ0041063', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0041007', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0041008', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0041009', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0041010', 4.0),
        ('WLdH1AfmPMaRsRLJZMdlR8gPJNm2', 'LV2PQ0052001', 4.0),
        ('lXHmbpmFA5ZXIkIX8MoU8sBpk7i2', 'LV2PQ0052001', 5.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV2PQ0052001', 5.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0052001', 5.0),
        ('eg93QctMN9ScQ7aJo040afqcor12', 'LV2PQ0052003', 5.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0052003', 5.0),
        ('WLdH1AfmPMaRsRLJZMdlR8gPJNm2', 'LV2PQ0052003', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0052004', 5.0),
        ('WLdH1AfmPMaRsRLJZMdlR8gPJNm2', 'LV2PQ0052004', 4.0),
        ('WLdH1AfmPMaRsRLJZMdlR8gPJNm2', 'LV2PQ0052027', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0052027', 5.0),
        ('WLdH1AfmPMaRsRLJZMdlR8gPJNm2', 'LV2PQ0052049', 4.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0052049', 5.0),
        ('XnnLfGhyiWdBQm0CFVxnp7Egcaw1', 'LV2PQ0052050', 4.0)]

data = pd.DataFrame(data, columns=col_names)
print(data)

# USER_ID, PRB_ID, LABEL 컬러만 추출하여 새로운 데이터프레임 생성
# data = df[['USER_ID', 'PRB_ID', 'label']]
# logger.info(f'필요한 컬럼만 추출한 상태 ::: {data}')
# logger.info(f'data ::: {data}')



# 유저 수
data['USER_ID'].nunique()
print("유저 수: ", data['USER_ID'].nunique())

# 문제 수
data['PRB_ID'].nunique()
print("문제 수: ", data['PRB_ID'].nunique())

# 데이터 인덱싱
user_unique = data['USER_ID'].unique()
quest_unique = data['PRB_ID'].unique()

# # 유저, 문제 indexing 하는 코드. idx는 index의 약자
user_to_idx = {v: k for k, v in enumerate(user_unique)}
quest_to_idx = {v: k for k, v in enumerate(quest_unique)}

# 인덱싱이 잘 되었는지 확인해 봅니다.
print(user_to_idx['eg93QctMN9ScQ7aJo040afqcor12'])  # 4명의 유저 중 처음 추가된 유저이니 0이 나와야 합니다.
print(quest_to_idx['LV1PQ0041059'])

# # user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다.
# # 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다.
temp_user_data = data['USER_ID'].map(user_to_idx.get).dropna()
if len(temp_user_data) == len(data):  # 모든 row가 정상적으로 인덱싱되었다면
    print('user_id column indexing OK!!')
    data['USER_ID'] = temp_user_data  # data['user_id']을 인덱싱된 Series로 교체해 줍니다.
else:
    print('user_id column indexing Fail!!')

# artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다.
temp_artist_data = data['PRB_ID'].map(quest_to_idx.get).dropna()
if len(temp_artist_data) == len(data):
    print('artist column indexing OK!!')
    data['PRB_ID'] = temp_artist_data
else:
    print('artist column indexing Fail!!')


num_user = data['USER_ID'].nunique()
num_artist = data['PRB_ID'].nunique()

print(data.USER_ID)
print()
print(data.PRB_ID)

csr_data = csr_matrix((data.LABEL, (data.USER_ID, data.PRB_ID)), shape=(num_user, num_artist))
print(f"csr_data ::: {csr_data}")


# implicit 라이브러리에서 권장하고 있는 부분
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'

# Implicit AlternatingLeastSquares 모델의 선언
als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False, iterations=15,
                                    dtype=np.float32)

# als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
csr_data_transpose = csr_data.T.tocsr()
print(f'csr_data_transpose ::: {csr_data_transpose}')

# 모델 훈련
als_model.fit(csr_data_transpose)

# logger.info('========= 모델 훈련 끝 =========')
#
# # 비슷한 문제 찾기
favorite_artist = 'LV1PQ0041015'
artist_id = quest_to_idx[favorite_artist]
print('quest_to_idx ::: ', quest_to_idx)
# print('artist_id ::: ', artist_id)
similar_artist = als_model.similar_items(artist_id, N=3)
print('similar_artist ::: ', similar_artist)

idx_to_artist = {v: k for k, v in quest_to_idx.items()}
temp = [idx_to_artist[i] for i in similar_artist[0]]
# print(temp)

user = user_to_idx['eg93QctMN9ScQ7aJo040afqcor12']
# recommend에서는 user*item CSR Matrix를 받습니다.
# print(als_model.recommend_all(user, 3))
print('csr_data_transpose[user] ::: ', csr_data_transpose[user])
artist_recommended = als_model.recommend(user, csr_data_transpose[user], N=3, filter_already_liked_items=True)
print('artist_recommended ::: ', artist_recommended)

# index to artist
print(*[idx_to_artist[i] for i in artist_recommended[0]])
#
# # 추천 기여도 확인
# rihanna = artist_to_idx['LV2PQ0052003']
# explain = als_model.explain(user, csr_data, itemid=rihanna)
#
# [(idx_to_artist[i[0]], i[1]) for i in explain[1]]
