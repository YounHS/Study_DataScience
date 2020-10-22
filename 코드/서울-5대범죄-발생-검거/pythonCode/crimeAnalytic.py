import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import json

import pydeck as pdk
import geopandas as gpd


# -------------------------------------------------------------------- folium
def func(x):
    d = {'count': x['건수'].sum()}
    return pd.Series(d, index=['count'])


def func2(x):
    d = {'diff': x['count'].diff()}
    return pd.Series(d, index=['diff'])


dataFrame = pd.read_csv("../yearInter.csv")

print(dataFrame)
print('----------------------------------------------')

# eachTopRank = dataFrame.describe(include=[np.object])
#
# print(eachTopRank)
# print('----------------------------------------------')
# 각 죄종별 범죄 발생/검거에 대한 상관관계를 구하고 싶다.
# 죄종별 발생/검거별 그룹화 필요
groupedDF = dataFrame.groupby(['발생검거', '죄종']).apply(func)
print(groupedDF)
print('----------------------------------------------')
resetGroupedDF = groupedDF.reset_index()
print(resetGroupedDF)
print('----------------------------------------------')
resetGroupedDF.to_csv('../resetGroupdDF.csv')

resetGroupedDF2 = pd.read_csv("../resetGroupdDF.csv")
groupedResetGroupedDF = resetGroupedDF2.groupby(['죄종'], sort=False).apply(func2)
print(groupedResetGroupedDF.reset_index())
print('----------------------------------------------')

# print(dataFrame.corr())
# print('----------------------------------------------')
#
# print(dataFrame.cov())
# print('----------------------------------------------')

# 죄종/발생검거 컬럼 합치기
dataFrame['죄종/발생검거'] = dataFrame['죄종'].astype(str) + ' ' + dataFrame['발생검거']
print(dataFrame)
print('----------------------------------------------')

# 지역별 죄종/발생검거 현황 컬럼으로 pivot 시키기 (날짜 컬럼은 사라지므로 시계열로서는 의미가 없어지며 지역의 특성에 집중됨)
# 시계열 데이터로 분석하기에는 애초에 2000 ~ 2018년까지 19개 뿐이므로 좋은 분석 결과를 얻기 힘들 것으로 사료됨
# pivot 함수 적용 시, index 중복 오류가 뜸. pd.pivot_table로 대체 가능
# pivotDataFrame = dataFrame.pivot(index='구분', columns='죄종/발생검거', values='건수')
pivotDataFrame = pd.pivot_table(dataFrame, index='구분', columns='죄종/발생검거', values='건수', aggfunc=np.sum)
print(pivotDataFrame.reset_index())
print('----------------------------------------------')
pivotDataFrame.reset_index().to_csv('../pivotDataFrame.csv', encoding='utf-8')

# 아무곳이나 좌표 찍어서 지도로 보여주기
# map_osm = folium.Map(location=[45.5236, -122.6750], zoom_start=12)
# map_osm.save('test.html')

# 서울 구역 json 데이터로 표시해주기
seoul_geo_path = './seoul_area.json'
geo_str = json.load(open(seoul_geo_path, encoding='utf-8'))

pivotData = pd.read_csv('../pivotDataFrame.csv')
pivotData['구분'] = pivotData['구분'] + '구'
pivotData['강간 검거율'] = pivotData['강간 검거'] / pivotData['강간 발생'] * 100
mapA = folium.Map(location=[37.5502, 126.982],
                  zoom_start=12,
                  tiles='Stamen Toner')
pivotData.set_index('구분')
print(pivotData)
print('----------------------------------------------')
print(pivotData.index)
folium.Choropleth(
    geo_data=geo_str,
    data=pivotData,
    columns=['구분', '강간 검거율'],
    fill_color='PuRd',
    key_on='feature.id').add_to(mapA)
folium.Marker([37.662751, 127.042490], tooltip='도봉구').add_to(mapA)
folium.Marker([37.639884, 127.012190], tooltip='강북구').add_to(mapA)
folium.Marker([37.650661, 127.076642], tooltip='노원구').add_to(mapA)
folium.Marker([37.601949, 127.020867], tooltip='성북구').add_to(mapA)
folium.Marker([37.599489, 127.100127], tooltip='중랑구').add_to(mapA)
folium.Marker([37.577270, 127.057688], tooltip='동대문구').add_to(mapA)
folium.Marker([37.583764, 127.976527], tooltip='종로구').add_to(mapA)
folium.Marker([37.593417, 127.923679], tooltip='은평구').add_to(mapA)
folium.Marker([37.569272, 127.938852], tooltip='서대문구').add_to(mapA)
folium.Marker([37.556764, 127.996630], tooltip='중구').add_to(mapA)
folium.Marker([37.551760, 127.049067], tooltip='성동구').add_to(mapA)
mapA.save('test2.html')

# -------------------------------------------------------------------- pydeck

# pdf = gpd.read_file(seoul_geo_path)
# folium.LayerControl().add_to(mapA)
# folium.Choropleth(
#     geo_data=geo_str,
#     data=pivotData,
#     columns=['구분', '강간 발생'],
#     fill_color='PuRd',
#     key_on='feature.id').add_to(mapA)
# mapA.save('test2.html')
