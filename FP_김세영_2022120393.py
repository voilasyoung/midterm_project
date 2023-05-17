#!/usr/bin/env python
# coding: utf-8

# # Kaggle Project: Salary prediction

# ## Describe my Dataset
# 
# ### URL : https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer
# 
# ### Task: 
#         1. 필요한 library를 import
#         2. 데이터셋 생성 및 분할
#         3. 모델 정의: DecisionTreeRegression & Neural Network
#         4. 각 모델 학습 및 검증
#         5. Test data를 통한 최종 성능 평가
#         
# 
# ### Datasets: 373개의 데이터를 train: validation: test = 6:2:2의 비율로 분할
#   * Train dataset: 238개(60%)
#   * Validation dataset: 60개(20%)
#   * Test dataset: 75개(20%)
# 
# ### Features(x): Age, Education Level, Job Title, Years of Experience
# 
# ### Target(y): Salary

# ## Import Library

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# ## Build Desicision Tree Regression Model

# #### Data preprocessing

# In[129]:


# 데이터 불러오기
salary_data = pd.read_csv('Salary Data.csv')


# In[130]:


salary_data.info()


# In[131]:


salary_data.head()


# In[132]:


# 데이터 결측치 확인
salary_data.isnull().sum()


# In[133]:


salary_data.shape


# In[134]:


# Gender, Job Title의 같은 항목들을 숫자로 변환

salary_data['Gender'] = salary_data['Gender'].replace({'Male':1, 'Female':0})

job_title_encoder = LabelEncoder()
salary_data['Job Title'] = job_title_encoder.fit_transform(salary_data['Job Title'])


# In[136]:


# 상관관계 분석
job_correlation = salary_data['Job Title'].corr(salary_data['Salary'])
years_correlation = salary_data['Scaled Years of Experience'].corr(salary_data['Salary'])
age_correlation = salary_data['Age'].corr(salary_data['Salary'])

print('상관관계: ', job_correlation)
print('상관관계: ', years_correlation)
print('상관관계: ', age_correlation)

# job title correlation이 너무 낮으므로 job title 삭제
salary_data = salary_data.drop('Job Title', axis=1)


# In[135]:


# years of experience 정규화

scaler = MinMaxScaler()

# nemeric 컬럼 뽑기 

years_of_experience = salary_data['Years of Experience']
years_of_experience

#2d 배열로 변환 후 정규화 
scaled_years_of_experience = scaler.fit_transform(years_of_experience.values.reshape(-1, 1))

# 데이터에 추가
salary_data['Scaled Years of Experience'] = scaled_years_of_experience

# Years of Experience column 삭제
salary_data = salary_data.drop('Years of Experience', axis=1)


# In[137]:


## age 정규화

# nemeric 컬럼 뽑기 
age = salary_data['Age']
age

#2d 배열로 변환 후 정규화 
scaled_age = scaler.fit_transform(age.values.reshape(-1, 1))

# 데이터에 추가
salary_data['Scaled_Age'] = scaled_age

# Age column 삭제
salary_data = salary_data.drop('Age', axis=1)


# In[138]:


# Education Level을 원핫 인코딩
education_level_encoded = pd.get_dummies(salary_data['Education Level'], prefix='Education')

# 인코딩 결과 데이터프레임에 반영
salary_data = pd.concat([salary_data, education_level_encoded], axis=1)

# 결과 확인
print(salary_data.head())

# Education Level column 삭제
salary_data = salary_data.drop('Education Level', axis=1)


# In[139]:


# 데이터 확인
salary_data


# #### Model Construction

# In[140]:


# test set, validation set, train set 설정 

Y = salary_data['Salary']
X = salary_data.drop(['Salary'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state=0)


# In[141]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)


# #### Train Model & Select Model

# In[149]:


model = DecisionTreeRegressor(max_depth= 5, max_features= 'sqrt', min_samples_leaf= 1)
model.fit(x_train, y_train)


# In[148]:


# grid search를 통한 hyperparameter 최적화

param_grid = {
    'max_depth': [3, 5, 7, 9, 100],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Grid Search 수행
grid_model = GridSearchCV(model, param_grid, cv=5)
grid_model.fit(x_train, y_train)

# 최적의 Hyperparameter 값 출력
print("최적의 Hyperparameter:", grid_model.best_params_)


# ## Performance

# In[150]:


# Train set에서의 예측값 계산
y_train_pred = model.predict(x_train)

# Train set에서의 R-squared 값 계산
r2_train = r2_score(y_train, y_train_pred)

# Validation set에서의 예측값 계산
y_val_pred = model.predict(x_val)

# Validation set에서의 R-squared 값 계산
r2_val = r2_score(y_val, y_val_pred)

# Test set에서의 예측값 계산
y_test_pred = model.predict(x_test)

# Test set에서의 R-squared 값 계산
r2_test = r2_score(y_test, y_test_pred)

print("Train set R-squared:", r2_train)
print("Validation set R-squared:", r2_val)
print("Test set R-squared:", r2_test)


# #### 중간 과제 Decision Tree performance
# ##### Train set R-squared: 0.9630213129532259
# ##### Validation set R-squared: 0.8125884097282398
# ##### Test set R-squared: 0.9061828247680328
# ##### 이전 모델 보다 performance가 향상되었다.

# ## Build Neural Network Model

# #### Data preprocessing

# In[152]:


salary_data = salary_data.reindex(['Gender', "Education_Bachelor's", "Education_Master's", 'Education_PhD', 'Scaled_Age', 'Scaled Years of Experience', 'Salary'], axis=1)
salary_data


# #### model construction

# In[154]:


train_data, test_data = train_test_split(salary_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

x_train = torch.Tensor(train_data.drop(['Salary'], axis=1).values)
y_train = torch.Tensor(train_data['Salary'].values)

x_val = torch.Tensor(val_data.drop(['Salary'], axis=1).values)
y_val = torch.Tensor(val_data['Salary'].values)

x_test = torch.Tensor(test_data.drop(['Salary'], axis=1).values)
y_test = torch.Tensor(test_data['Salary'].values)


# In[155]:


print(x_train.shape)
print(y_train.shape)
print(x_val.shape)


# #### Train Model & Select Model

# In[175]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 120)
        self.fc2 = nn.Linear(120, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# 모델 학습
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

train_losses = []
val_losses = []

num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x_train)
    train_loss = criterion(output, y_train.unsqueeze(1))
    train_losses.append(train_loss.item())
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_output = model(x_val)
        val_loss = criterion(val_output, y_val.unsqueeze(1))
        val_losses.append(val_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")


# ## Performance

# In[176]:


# 모델 평가
with torch.no_grad():
    train_output = model(x_train)
    train_mse = nn.functional.mse_loss(train_output, y_train.unsqueeze(1)).item()
    train_r2 = r2_score(y_train, train_output.numpy().flatten())

    val_output = model(x_val)
    val_mse = nn.functional.mse_loss(val_output, y_val.unsqueeze(1)).item()
    val_r2 = r2_score(y_val, val_output.numpy().flatten())

    test_output = model(x_test)
    test_mse = nn.functional.mse_loss(test_output, y_test.unsqueeze(1)).item()
    test_r2 = r2_score(y_test, test_output.numpy().flatten())

print("----------------------------------------------------------")    
print(f"Train MSE: {train_mse}, Train R^2: {train_r2}")
print(f"Validation MSE: {val_mse}, Validation R^2: {val_r2}")
print(f"Test MSE: {test_mse}, Test R^2: {test_r2}")


# #### 중간 과제 neural network performance
# ##### Train MSE: 408524224.0, Train R^2: 0.8202534378581079
# ##### Validation MSE: 435900032.0, Validation R^2: 0.8134891012092795
# ##### Test MSE: 372709216.0, Test R^2: 0.844547898493238
# ##### performance가 향상되었다.
