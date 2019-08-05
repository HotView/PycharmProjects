import tensorflow as tf
## 模型的保存
save_path ='...'
saver = tf.train.Saver()
sess = tf.Session()
saver.save(sess,save_path)

## 模型的恢复
save_path = ".."
saver = tf.train.Saver()
sess= tf.Session()
saver.restore(sess,save_path)
## 多次模型的保存和恢复
save_path = ".."
saver = tf.train.Saver()
sess= tf.Session()
epoch = 5
n =None
if epoch%n==0:
    saver.save(sess,save_path,global_step=epoch)
## 恢复最新的模型
save_path = ".."
model = tf.train.latest_checkpoint(save_path)
saver = tf.train.Saver()
sess= tf.Session()
saver.restore(sess,model)
