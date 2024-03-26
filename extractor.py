# import ptvsd
# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('172.18.30.113', 6666))
# # Pause the program until a remote debugger is attached
# print('wait for attach')
# ptvsd.wait_for_attach()
# print('succeed')

import os

def reshape_list(old_list, group=3):
    new_list = []
    for i in range(group):
        new_list.append([])
        for j in range(len(old_list)):
            if j % group == i:
                new_list[i].append(old_list[j])

    return new_list

# 1 打开文件测试
#size = 2048
list_test_1 = []
list_test_2 = []
list_test_3 = []
list_test_4 = []
list_test_5 = []
list_avg = []
list_na = []
list_stdp = []
#new_list = []
#with open('D:\\SNN\\chengxiang-data\\nohup_n_0.6_s_0.1', 'r', encoding='UTF-8') as f:
    #lines = f.readlines()
    #for line in lines:
        #print(len(str(line)))
        #if str(line)[0] == '[' or str(line)[0] == 'a':
            #new_list.append(line)
    #print(new_list)
    #print(f.read(size))

# 2 单个文件处理
find0 = 'EVALUATION RESULTS:'
find1 = "[0, 1] "
find2 = "[2, 3] "
find3 = "[4, 5] "
find4 = "[6, 7] "
find5 = "[8, 9] "
find6 = "average"
find7 = "na: "
find8 = 'stdp: '

# with open('D:\\SNN\\chengxiang-data\\nohup_n_0.6_s_0.1', 'r', encoding='UTF-8') as f:
with open('./recorded_nohup/nohup_n_0.6_s_0.1', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        zero = str(line).find(find0)
        one = str(line).find(find1)
        two = str(line).find(find2)
        three = str(line).find(find3)
        four = str(line).find(find4)
        five = str(line).find(find5)
        six = str(line).find(find6)
        seven = str(line).find(find7)
        eight = str(line).find(find8)

        if zero != -1:
            break

        if one != -1:
            list_test_1.append(str(line)[one+9:one+27])
        elif two != -1:
            list_test_2.append(str(line)[two+9:two+27])
        elif three != -1:
            list_test_3.append(str(line)[three+9:three+27])
        elif four != -1:
            list_test_4.append(str(line)[four+9:four+27])
        elif five != -1:
            list_test_5.append(str(line)[five+9:five+27])
        elif six != -1:
            list_avg.append(str(line)[six+9:six+27])

        if seven != -1:
            if str(line)[seven + 11] == '1':
                list_na.append(str(line)[seven + 11:seven + 13])
            else:
                list_na.append(str(line)[seven + 11:seven + 17])
        if eight != -1:
            if str(line)[eight + 13] == '1':
                list_stdp.append(str(line)[eight + 13:eight + 15])
            else:
                list_stdp.append(str(line)[eight + 13:eight + 19])

    list_na = reshape_list(list_na)
    list_stdp = reshape_list(list_stdp)

    print('list_test_1', list_test_1)
    print('list_test_2', list_test_2)
    print('list_test_3', list_test_3)
    print('list_test_4', list_test_4)
    print('list_test_5', list_test_5)
    print('list_avg', list_avg)
    print('list_na', list_na)
    print('list_stdp', list_stdp)

# 3 批次处理
# file_dir = 'D:\\SNN\\chengxiang-data'
file_dir = './recorded_nohup'
list_out = []
file_num = 0

# for (root, dirs, files) in os.walk(file_dir):
#     for file_name in files:
#         file_num = file_num + 1
#         with open(os.path.join(root, file_name), 'r', encoding='UTF-8') as f_test:
#             line = f_test.readlines()
#             for line in lines:
#                 one = str(line).find(find1)
#                 two = str(line).find(find2)
#                 three = str(line).find(find3)
#                 four = str(line).find(find4)
#                 five = str(line).find(find5)
#                 six = str(line).find(find6)
#                 if one != -1:
#                     list_out.append(str(line)[one + 9:one + 27])
#                 elif two != -1:
#                     list_out.append(str(line)[two + 9:two + 27])
#                 elif three != -1:
#                     list_out.append(str(line)[three + 9:three + 27])
#                 elif four != -1:
#                     list_out.append(str(line)[four + 9:four + 27])
#                 elif five != -1:
#                     list_out.append(str(line)[five + 9:five + 27])
#                 elif six != -1:
#                     list_out.append(str(line)[six + 9:six + 27])
#     print('list_out', list_out)
#     data_out = list(map(float, list_out))
#     print('data_out', data_out)                   # data_out 就是遍历所有文件 按照顺序把accuracy转为float之后放入







