
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



"""
完成对训练集图像长宽、类别数量信息的统计
"""
train_label_path = "train_new/train/train1A_new.csv"
# train_img_path = 'C:/code/hehuang_cup/train/train1A/*.jpg'

csv_data = pd.read_csv(train_label_path)
csv_data = csv_data.fillna(.0)

print(csv_data.iloc[0, 0])

# print(csv_data.columns.tolist())

info = dict() # 统计各个属性的数量

for col_name in csv_data.columns.tolist():
	if col_name == 'name':
		continue
	if col_name in ['upperLength', 'clothesStyles', 'hairStyles']:
		attri_name = set(csv_data[col_name].to_list()) # 获取属性列表
		
		info.update(csv_data[col_name].value_counts().to_dict())
	else:
		# idx = csv_data[col_name] != .0
		# csv_data[col_name][idx] = 1.
		info[col_name] = csv_data[col_name].sum()

print(info.keys())

plt.figure()
plt.xlabel('attribute name')
plt.ylabel('counts')
plt.grid()
for i, (k, v) in enumerate(info.items()):
	# print(i)
	plt.bar(i+1, v, width=0.3)
	plt.text(i+0.75, v+0.5, str(round(v, 1)))

plt.xticks(range(21), info.keys(), rotation=45)

plt.show()

print(info)



# ### 修改最后的结果为全0，看有没有影响
# result_path = 'results/result_aug_mix_resample_v2.csv'
# data = pd.read_csv(result_path)
# data.iloc[:, 4:] = np.zeros([1500, 11])
# data.to_csv('result.csv', index=None)
