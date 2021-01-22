_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=10,
    )
)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

data_root = '/Users/hzy/Desktop/天池/零基础入门CV - 街景字符编码识别/'
data = dict(
    train=dict(
        ann_file=data_root + 'coco/train.json',
        img_prefix=data_root + 'mchar_train/',),
    val=dict(
        ann_file=data_root + 'coco/val.json',
        img_prefix=data_root + 'mchar_val/',),
    test=dict(
        ann_file=data_root + 'coco/val.json',
        img_prefix=data_root + 'mchar_val/',))