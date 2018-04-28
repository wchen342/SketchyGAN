import os
import sys

sys.path.append('.')
sys.path.append('./pycocotools')
import pycocotools.coco as coco

pascal_classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
                  'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
                  'dining table', 'potted plant', 'sofa', 'tv/monitor']
pascal_classes_mapped = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane',
                         'bicycle', 'boat', 'bus', 'car (sedan)', 'motorcycle', 'train', 'bottle', 'chair',
                         'dining table', 'potted plant', 'sofa', 'tv']

images_dir = '../Datasets/COCO/coco-master/images'
anno_dir = '../Datasets/COCO/coco-master/annotations'
itrain_2014 = 'instances_train2014.json'
ival_2014 = 'instances_val2014.json'


def img_info_list_to_dict(input_list):
    dic = {}
    for i in input_list:
        image_id = i['id']
        assert image_id not in dic.keys()
        dic[image_id] = i
    return dic


def get_all_images_data_categories(split, catIds=[]):
    if split == 'train':
        COCO = coco.COCO(annotation_file=os.path.join(anno_dir, itrain_2014))
    elif split == 'test':
        COCO = coco.COCO(annotation_file=os.path.join(anno_dir, ival_2014))
    if len(catIds) == 0:
        return COCO.loadImgs(COCO.getImgIds()), COCO.loadAnns(COCO.getAnnIds()), COCO.loadCats(COCO.getCatIds())
    else:
        return COCO.loadImgs(ids=COCO.getImgIds(catIds=catIds)), COCO.loadAnns(
            ids=COCO.getAnnIds(catIds=catIds)), COCO.loadCats(COCO.getCatIds(catIds=catIds))


def expand_bbox(bbox, max_height, max_width, frac):
    assert len(bbox) == 4
    half_width = round(bbox[2] / 2)
    half_height = round(bbox[3] / 2)
    mid_x = bbox[0] + half_width
    mid_y = bbox[1] + half_height

    x_min = max(0, mid_x - half_width * frac)
    y_min = max(0, mid_y - half_height * frac)
    x_max = min(max_width, mid_x + half_width * frac)
    y_max = min(max_height, mid_y + half_height * frac)
    return [round(x_min), round(y_min), round(x_max), round(y_max)]


def get_shared_classes(input=None, print_out=False, output_file=True):
    if input is None:
        ret = get_all_images_data_categories('train')[2]
    else:
        ret = input
    coco_classes = [cls['name'] for cls in ret]
    class_dict = {item['name']: item for item in ret}
    # Convert 'car' to 'car (sedan)' for comparison with Sketchy
    coco_classes[coco_classes.index('car')] = 'car (sedan)'
    with open('../../shared_classes', 'r') as f:
        shared_classes = [cls[:-1].replace('_', ' ') for cls in f.readlines()]

    if print_out:
        print(len(shared_classes))
        print(shared_classes)
    if output_file:
        shared_classes2 = [(cls + '\n') for cls in shared_classes if cls in coco_classes]
        with open('../../shared_classes2', 'w') as f:
            f.writelines(shared_classes2)

    shared_classes = [cls for cls in shared_classes if cls in coco_classes]
    print([cls for cls in shared_classes if cls in coco_classes and cls in pascal_classes_mapped])
    shared_classes[shared_classes.index('car (sedan)')] = 'car'  # Convert 'car' back
    output_dict = {class_dict[cls]['id']: class_dict[cls] for cls in shared_classes}
    return shared_classes, output_dict


def get_bbox():
    img_info, seg_info, cat_info = get_all_images_data_categories('train')
    img_info = img_info_list_to_dict(img_info)
    shared_classes, cls_info = get_shared_classes(input=cat_info, print_out=False, output_file=False)
    bbox_list = {cls: [] for cls in shared_classes}
    for object in seg_info:
        category_id = object['category_id']
        if category_id not in cls_info:
            continue

        category_name = cls_info[category_id]['name']
        bbox = object['bbox']
        image_id = object['image_id']
        iscrowd = object['iscrowd']

        this_img_info = img_info[image_id]
        file_name = this_img_info['file_name']
        height = this_img_info['height']
        width = this_img_info['width']
        bbox = expand_bbox(bbox, height, width, 1.5)

        bbox_list[category_name].append({
            'image_id': image_id,
            'category_name': category_name,
            'category_id': category_id,
            'iscrowd': iscrowd,
            'file_name': file_name,
            'height': height,
            'width': width,
            'bbox': bbox,
        })

    return bbox_list


if __name__ == '__main__':
    get_bbox()
