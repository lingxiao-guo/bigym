import pyrosbag

# 打开bag文件
bag = pyrosbag.Bag('rosbag_demo/1-0000-20250105151909.bag')

# 读取消息
for topic, msg, t in bag.read_messages():
        print(f"Topic: {topic}, Timestamp: {t.to_sec()}")

# 关闭bag文件
bag.close()