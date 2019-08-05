import cv2
import os
import numpy as np
import tensorflow as tf
def rebuild(dir):
    for root,dirs,files in os.walk(dir):
        for file in files:
            filepath =  os.path.join(root,file)
            try:
                image = cv2.imread(filepath)
                dim = (227,227)
                resized = cv2.resize(image,dim)
                path = "E:/cat_and_dog/train2/"+file
                print(path)
                cv2.imwrite(path,resized)
                print(filepath)
            except:
                print("Error:"+filepath)
                os.remove(filepath)
        cv2.waitKey(2000)

def get_file(file_dir):
    images = []
    temp = []
    for root,sub_folders,files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root,name))
        for name in sub_folders:
            temp.append(os.path.join(root,name))

        # print(root,"root")
        # print(sub_folders,"sub_dirs")
        # print(files,"files")
    print("++++++++++++")
    print(temp)
    print("++++++++++++")
    print("------------")
    print(images)
    print("------------")
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split("\\")[-1]
        if letter=='cat':
            labels = np.append(labels,n_img*[0])
        else:
            labels = np.append(labels,n_img*[1])
    # shuffle
    temp = np.array([images,labels])
    print(temp.shape)
    temp = temp.transpose()
    print(temp.shape)
    np.random.shuffle(temp)
    print(temp.shape)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

def get_batch(image_list,label_list,img_width,img_height,batch_size,capacity):
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image = tf.image.per_image_standardization(image)#图片标准化
    image_batch,label_batch =tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
# rebuild("E:/cat_and_dog/train")