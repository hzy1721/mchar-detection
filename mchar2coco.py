import os.path as osp
import mmcv


def convert_mchar_to_coco(ann_file, out_file, image_prefix):
    """mchar格式 -> COCO格式
    参数：
        ann_file: mchar标注文件路径
        out_file: 输出的COCO标注文件路径
        image_prefix: 除文件名外的图片路径前缀
    """
    # 加载JSON文件为字典
    data_infos = mmcv.load(ann_file)

    # annotations: [annotation]
    annotations = []
    # images: [image]
    images = []
    # 物体数量
    obj_count = 0
    # 遍历每张图片的信息
    idx = 0
    for k, v in mmcv.track_iter_progress(data_infos.items()):
        # 文件名
        filename = k
        # 文件路径
        img_path = osp.join(image_prefix, filename)
        # 读取图片的高和宽
        height, width = mmcv.imread(img_path).shape[:2]

        # 添加一个image到images
        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        x1_list = v['left']
        y1_list = v['top']
        w_list = v['width']
        h_list = v['height']
        l_list = v['label']

        length = len(x1_list)

        # 遍历每个物体
        for iidx in range(length):
            # bbox
            x1, y1 = x1_list[iidx], y1_list[iidx]
            w, h = w_list[iidx], h_list[iidx]
            label = l_list[iidx]
            # annotation
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=label,
                bbox=[x1, y1, w, h],
                area=w * h,
                segmentation=[],
                iscrowd=0)
            # 添加一个annotation到annotations
            annotations.append(data_anno)
            # 物体数量+1
            obj_count += 1

        idx += 1

    category_list = []
    for id in range(10):
        category_list.append({'id': id, 'name': str(id)})

    # 最外层JSON
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        # 只有一个类：balloon
        categories=category_list)
    # 写入到文件
    mmcv.dump(coco_format_json, out_file)


convert_mchar_to_coco('/Users/hzy/Desktop/天池/零基础入门CV - 街景字符编码识别/mchar_train.json', 'coco/train.json',
                      '/Users/hzy/Desktop/天池/零基础入门CV - 街景字符编码识别/mchar_train')
convert_mchar_to_coco('/Users/hzy/Desktop/天池/零基础入门CV - 街景字符编码识别/mchar_val.json', 'coco/val.json',
                      '/Users/hzy/Desktop/天池/零基础入门CV - 街景字符编码识别/mchar_val')