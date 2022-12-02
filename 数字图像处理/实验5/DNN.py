# import os
#
#
# def change_name(xlsbpath, name):
#     # xlsbpath = r"D:\2020185 _and_10208\Kernel_lessons\数字图像处理\实验5\dataset\su"
#     os.chdir(xlsbpath)  # 更改当前路径
#     filelist = os.listdir(xlsbpath)  # 该文件夹下所有的文件（包括文件夹）
#     print(filelist)
#     j = 0
#     for i in filelist:
#         os.rename(i, str(j) + '_' + name + '.jpg')
#         j += 1
#
#
# if __name__ == '__main__':
#     path = 'C:\\Users\\shulin\\Desktop\\dataset'
#     os.chdir(path)
#     filelist = os.listdir(path)
#
#     for i in filelist:
#         # print(i)
#         # print(path + '\\' + i)
#         change_name(path + '\\' + i, i)
#     print('Finished!')
import cv2
path = './work/license_plate.jpg'
license_plate = cv2.imread(path)
print(license_plate)