import cv2

fps = 10
size = (1600, 600)
output_path = "/Users/lihaobo/PycharmProjects/ENV_seq2seq/figs_aqms_72to1/video.avi"
video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
for i in range(100, 600):
    image_path = "/Users/lihaobo/PycharmProjects/ENV_seq2seq/figs_aqms_72to1/a%i.png" % i
    print(i)
    img = cv2.imread(image_path)
    org = (750, 20)
    color = (255, 255, 255)
    img = cv2.putText(img, 'OpenCV', org, color, cv2.LINE_AA)
    video.write(img)

video.release()
cv2.destroyAllWindows()