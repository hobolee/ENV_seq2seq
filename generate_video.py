import cv2
import datetime

fps = 10
size = (1600, 600)
output_path = "/Users/lihaobo/PycharmProjects/ENV_seq2seq/figs/diff_after_cor/video.mp4"

#MP4, size is much lower than avi
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(output_path, fourcc, fps, size)
#AVI
# video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

date = datetime.datetime(2021, 12, 1, 23)
delta = datetime.timedelta(hours=1)
for i in range(2626-30*24, 2626):
    image_path = "/Users/lihaobo/PycharmProjects/ENV_seq2seq/figs/diff_after_cor/a%i.png" % i
    date = date + delta
    print(i)
    img = cv2.imread(image_path)
    org = (30, 55)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    img = cv2.putText(img, str(date)[:16], org, font, fontScale, color, thickness, cv2.LINE_AA)
    # org = (750, 20)
    # color = (255, 255, 255)
    # img = cv2.putText(img, 'OpenCV', org, color, cv2.LINE_AA)
    video.write(img)

video.release()
cv2.destroyAllWindows()