
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
from tensorflow.keras.models import Model

class DataLoad:


    def __init__(self):
        base_model = VGG16(weights='imagenet')  # 모델 가중치 학습
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        

    def _Feature_Extract(self, img): # 이미지 특징 추출 함수
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    
    
    def _Pokemon_Name(self): # 포켓몬 이름 추출 함수
        pokemon_name = pd.read_csv("./data/pokemon.csv")
        pokemon_name_list = list(pokemon_name['Name'])
        return pokemon_name_list
    

    def _Feature_Save(self, pokemon_name_list): # 포켓몬 특징 정보를 저장하는 함수
        features = []
        img_paths = []

        for i in pokemon_name_list:
            try:
                image_path = "./data/images/" + i + ".png"
                img_paths.append(image_path)

                # Extract Features
                feature = self._Feature_Extract(img=Image.open(image_path))

                features.append(feature)
            except:
                image_path = "./data/images/" + i + ".jpg"
                img_paths.append(image_path)

                # Extract Features
                feature = self._Feature_Extract(img=Image.open(image_path))

                features.append(feature)
                
        return features, img_paths

    
    
