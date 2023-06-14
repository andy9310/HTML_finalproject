import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertConfig, get_linear_schedule_with_warmup
import tensorflow as tf
import torch
import torch.nn as nn
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer

label_feature = 'Danceability'
numerical_features = ['Energy', 'Speechiness', 'Acousticness',
                    'Instrumentalness', 'Liveness', 'Valence', 'Tempo' ]
categorical_features = ['Album_type', 'Key', 'Licensed', 'official_video']
power_transform_features = ['Loudness', 'Duration_ms', 'Stream', 'Views', 'Likes', 'Comments']
string_features = [
    'Track', 'Artist', 'Composer', 'Album', 'Title', 'Channel', 'Description']

def data_loader( path ):
    train_ds = pd.read_csv( path )
    
    string_features = [
        'Track', 'Artist', 'Composer', 'Album', 'Title', 'Channel', 'Description'
    ]
    data_inputs = train_ds[ string_features  ]
    data_outputs = train_ds[ ['Danceability'] ]
    return data_inputs, data_outputs


def concate_data(data):
  string_features = [
        'Track', 'Artist', 'Composer', 'Album', 'Title', 'Channel', 'Description'
    ]
  text = ''
  for i in string_features:
    if pd.isnull(data[i]):
      text = text + ''
    else:
      text = text + data[i]+" "
  return text

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_text, input_label,tokenizer):
        self.labels = [ input_label.iloc[i]['Danceability'] for i in range(input_label.shape[0]) ]
        
        self.texts = [tokenizer(one_text, 
                                padding='max_length', 
                                max_length = 100, 
                                truncation=True,
                                return_tensors="pt") 
                      for one_text in input_text]
        print(self.texts[0])
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
    


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        #final_layer = self.relu(linear_output)
        return linear_output

def train(model, trains, vals, learning_rate, epochs):
    train, val = trains, vals
    # DataLoader根據batch_size獲取數據，訓練時選擇打亂樣本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=100)
  # 判斷是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device( 'cuda' if use_cuda else "cpu")
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 開始進入訓練循環
    for epoch_num in range(epochs):
      # 定義兩個變量，用於存儲訓練集的準確率和損失
            total_acc_train = 0
            total_loss_train = 0
      # 進度條函數tqdm
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.type(torch.LongTensor)
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通過模型得到輸出
                output = model(input_id, mask)
                #print(output)
                # 計算損失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 計算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
        # 模型更新
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # ------ 驗證模型 -----------
            # 定義兩個變量，用於存儲驗證集的準確率和損失
            total_acc_val = 0
            total_loss_val = 0
      # 不需要計算梯度
            with torch.no_grad():
                # 循環獲取數據集，並用訓練好的模型進行驗證
                for val_input, val_label in val_dataloader:
          # 如果有GPU，則使用GPU，接下來的操作同訓練
                    val_label = val_label.type(torch.LongTensor)
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train): .3f} 
              | Train Accuracy: {total_acc_train / len(train): .3f} 
              | Val Loss: {total_loss_val / len(val): .3f} 
              | Val Accuracy: {total_acc_val / len(val): .3f}''')

def evaluate(model, test_data):

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=5)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    outputs = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              outputs.append(output)
              print(output)
              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')    
    outputs = np.array(outputs)
    return outputs
    
    
def transform_function():

    normal_feature_transformer = MinMaxScaler
    skewed_feature_transformer = PowerTransformer
    imputer = SimpleImputer(strategy='mean')

    #######################################################################
    # load data
    train_ds = pd.read_csv( './train.csv' )
    test_ds = pd.read_csv( './test.csv' )

    #######################################################################
    # numerical , categorical
    # abs numerical features
    train_ds[numerical_features] = train_ds[numerical_features].abs()

    o_df = train_ds.copy()
    test_df = test_ds.copy()
    o_df[numerical_features] = normal_feature_transformer.fit_transform(train_ds[numerical_features])
    test_df[numerical_features] = normal_feature_transformer.fit_transform(test_ds[numerical_features])

    o_df[power_transform_features] = skewed_feature_transformer.fit_transform(train_ds[power_transform_features])
    test_df[power_transform_features] = skewed_feature_transformer.fit_transform(test_ds[power_transform_features])

    o_df = pd.get_dummies(o_df, columns=categorical_features)
    test_df = pd.get_dummies(test_df, columns=categorical_features)

    o_df = o_df.astype(float)
    test_df = test_df.astype(float)

    # for each numerical features, add a new column to indicate whether it is missing
    for feature in numerical_features:
        o_df[feature + '_missing'] = train_ds[feature].isna().astype(float)
    for feature in numerical_features:
        test_df[feature + '_missing'] = test_ds[feature].isna().astype(float)
    # replace all Nan to 0
    o_df = pd.DataFrame(imputer.fit_transform(o_df), columns=o_df.columns)
    test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

    label_df = train_ds.copy()
    label_df = pd.DataFrame(normal_feature_transformer.fit_transform(label_df[[label_feature]]), columns=[label_feature])

    train_X_df = o_df
    train_Y_df = label_df
    test_X_df = test_df

    train_X = train_X_df.to_numpy()
    train_Y = train_Y_df.to_numpy()
    test_X = test_X_df.to_numpy()
    feature_size = train_X.shape[1]
    output_size = 1


    
    config={ "pretrained_model":"bert-base-cased" }
    tokenizer = BertTokenizerFast.from_pretrained( config["pretrained_model"] )
    np.random.seed(112)
    input , output = data_loader('./train.csv')
    test_input, test_output = data_loader('./test.csv')
    test_input = test_input.to_numpy()
    test_output = test_output.to_numpy()
    input_train, input_val, input_test = np.split( input.sample(frac=1, random_state=42), [int(.8*len(input)), int(.9*len(input))])
    output_train, output_val, output_test = np.split( output.sample(frac=1, random_state=42), [int(.8*len(output)), int(.9*len(output))])
    input_train_text = [ concate_data(input_train.iloc[i]) for i in range(input_train.shape[0]) ]
    input_val_text = [ concate_data(input_val.iloc[i]) for i in range(input_val.shape[0]) ]
    input_test_text = [ concate_data(input_test.iloc[i]) for i in range(input_test.shape[0]) ]
    test_input_text = [ concate_data(test_input.iloc[i]) for i in range(test_input.shape[0]) ]
    print(len(input_val_text))
    train_i = Dataset(input_train_text,output_train,tokenizer)
    val_i = Dataset(input_val_text,output_val,tokenizer)
    test_i = Dataset(input_test_text,output_test,tokenizer)
    CUDA_LAUNCH_BLOCKING=1
    EPOCHS = 20
    model = BertClassifier()
    LR = 1e-6
    train(model, train_i, val_i, LR, EPOCHS)
    eva = evaluate(model,test_i )
    final_i = Dataset(test_input_text, test_output,tokenizer)
    label_transform_data = evaluate(model,final_i)
    train_X = np.append(train_X,label_transform_data,axis=1)
    
    return train_X, train_Y

