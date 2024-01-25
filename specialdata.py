import cv2
import numpy as np
import os
import mysql.connector
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import pickle
import openpyxl

def get_values(sheet):
    arr = []  
    for row in sheet.iter_rows(min_row=2, values_only=True):
        arr.append(list(row))
    return arr

# 資料庫初始化
conn = mysql.connector.connect(
    host='127.0.0.1',
    user='root',
    password='daisy218',
    database='data'
)

# 建立游標
cursor = conn.cursor()

table_name = "special_list"

# 刪除資料表（如果存在的話）
try:
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    print(f"{table_name} delete successfully")
except mysql.connector.Error as err:
    print(f"error：{err}")

# 執行 CREATE TABLE 語句來建立新的資料表
create_table_query = """ CREATE TABLE special_list (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    number VARCHAR(255),
    feature1 BLOB,
    img_path1 VARCHAR(255),
    feature2 BLOB,
    img_path2 VARCHAR(255),
    feature3 BLOB,
    img_path3 VARCHAR(255),
    feature4 BLOB,
    img_path4 VARCHAR(255),
    feature5 BLOB,
    img_path5 VARCHAR(255),
    feature6 BLOB,
    img_path6 VARCHAR(255)
) """

cursor.execute(create_table_query)
conn.commit()

# 載入模型
face_detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Excel路徑
excel = openpyxl.load_workbook('./special_list/special_list.xlsx')
s1 = excel['工作表1']
data = get_values(s1)

for space in data:
    # 創建一個用於存儲每個人的特徵和圖像路徑的字典
    person_data = {
        'name': space[1],
        'number': space[0],
        'feature1': None,
        'img_path1': space[2],
        'feature2': None,
        'img_path2': space[3],
        'feature3': None,
        'img_path3': space[4],
        'feature4': None,
        'img_path4': space[5],
        'feature5': None,
        'img_path5': space[6],
        'feature6': None,
        'img_path6': space[7]
    }

    for i, j in zip(range(2, 8), range(1, 7)):
        img_path = space[i]

        # 檢查是否存在
        if not os.path.exists(img_path):
            # 如果不存在，使用第一張照片填充
            img_path = os.path.join(space[2])

        image = cv2.imread(img_path)

        # 使用MTCNN檢測人臉
        faces = face_detector.detect(image)

        if len(faces) == 1:
            face = faces[0]  # 假設每張照片只有一個人臉
            x, y, w, h = face['box']
            face_image = image[y:y+h, x:x+w]
        else:
            face_image = image

        # 將人臉圖像轉換為FaceNet所需的格式
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_image, (160, 160))
        face_image = (face_image - 127.5) / 128.0  # 歸一化

        # 提取人臉特徵
        face_tensor = np.expand_dims(face_image.transpose(2, 0, 1), axis=0)
        face_tensor = torch.from_numpy(face_tensor).float()
        features = facenet(face_tensor)

        # 更新字典中的特徵和圖像路徑
        person_data[f'feature{j}'] = pickle.dumps(features.detach().numpy())

    # 將字典中的數據插入到數據庫中
    insert_query = f"INSERT INTO {table_name} ("
    for key in person_data.keys():
        insert_query += key + ","
    insert_query = insert_query[:-1] + ") VALUES ("
    for key in person_data.keys():
        insert_query += "%s,"
    insert_query = insert_query[:-1] + ")"
    values = tuple(person_data.values())
    cursor.execute(insert_query, values)
    conn.commit()

# 關閉資料庫連線
cursor.close()
conn.close()
