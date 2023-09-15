from PIL import Image
import click
import sys

from load import DataLoad
dl = DataLoad()
from model import Models
ml = Models()

@click.command()
@click.option('-n', '--pokemone-name', type = click.STRING, default = 'pikachu', help = '기준이 되는 포켓몬 이름 입력')
@click.option('-m', '--model-name', type = click.STRING, default = 'cos_score', help = '유사도 기법 선택')
def start_search(pokemone_name, model_name):

    # pokemon.csv 파일을 이용해 포켓몬들 이름을 리스트에 저장
    pokemon_name_list = dl._Pokemon_Name()
    # 특징 저장
    features, img_paths = dl._Feature_Save(pokemon_name_list)
    # 기준 이미지 불러오기
    img = Image.open(f"./data/images/{pokemone_name}.png")
    # 기준 이미지 특징 불러오기
    query = dl._Feature_Extract(img)


    # 이미지 유사도 모델 실행
    if model_name == "L2_norm":
        ml.L2_norm(query, features, img_paths)

    elif model_name == "cos_score":
        ml.cos_score(query, features, img_paths)

    else:
        print("해당하는 유사도 측정 기법이 없습니다.")
    
    print('<<<<<  END SEARCH  >>>>>')
    print(f"\tInput Value : pokemone-name == '{pokemone_name}' ")
    print(f"\tInput Value : model-name == '{model_name}' ")

    sys.exit(1) # 정상적으로 종료


if __name__ == '__main__':
    start_search()