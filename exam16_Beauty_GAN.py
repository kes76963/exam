import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() #뷰티gan이 tf v1에서 만들어진 거라서
import numpy as np

def align_faces(img, detector, sp) : #이미지 정렬 함수
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets :
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces

detector = dlib.get_frontal_face_detector() #얼굴만 찾아줌
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

#테스트 이미지
"""
test_img = dlib.load_rgb_image('./imgs/02.jpg')
test_faces = align_faces(test_img, detector, sp)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)

plt.show()
"""

#모델 load
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

#데이터 전처리
def preprocess(img):
    return(img/255. - 0.5) * 2 # -1 에서 1 사이
def deproess(img):
    return (img + 1 ) / 2 # 0~1 사이 값, 원래값으로 복원원
img1 = dlib.load_rgb_image('./imgs/12.jpg')
img1_faces = align_faces(img1, detector, sp)

img2 = dlib.load_rgb_image('./imgs/makeup/XMY-014.png')
img2_faces = align_faces(img2, detector, sp)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

#화장 입히기
src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0) #reshape, 마지막 한 차원 늘리기

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict = {X:X_img, Y:Y_img})
output_img = deproess(output[0])

fig, axes = plt.subplots(1,3,figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()

#얼굴 인식
"""

img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()

img_result = img.copy()
dets = detector(img) #얼굴의 좌표를 만들어준다
if len(dets) == 0 :
    print('cannot find faces!')
else :
    fig, ax = plt.subplots(1, figsize=(16,10))
    for det in dets :
        x, y, w, h = det.left(), det.top(), det.width(), det.height() #왼쪽, 위, 폭, 높이 좌표가 있음
        rect = patches.Rectangle((x,y),w,h, linewidth=2, edgecolor='r', facecolor='none') # 네모 그렸을 때 선 두께, 색, 내부 채우기 색
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets :
    s = sp(img,detection) #학습된 모델
    objs.append(s)
    for point in s.parts() :
        circle = patches.Circle((point.x, point.y),radius=3, edgecolor='r',facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
#plt.show()

faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) #패딩이 얼굴 부분 확대 0이면 얼굴만 보여줌
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)

for i, face in enumerate(faces) :
    axes[i+1].imshow(face)
plt.show()
"""