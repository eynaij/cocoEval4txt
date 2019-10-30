import json
import argparse
import numpy as np
from common.dataset.coco_eval import COCOEval

def parse_args():
    parser = argparse.ArgumentParser(description='Eval detection results in TXT format using pycocotools')
    parser.add_argument('anno_file', help='Absolute path of the annotation file(json)')
    parser.add_argument('txt_file', help='Absolute path of the txt file')
    parser.add_argument('category', 
                        help='Category of detection results, default category category_id:person 1 head 2 face 3, if necessary, change the main function corresponding to the gt,')
    args = parser.parse_args()
    return args

def txt2json(anno_file, txt_file ):
    print anno_file
    print txt_file

    fileName = []
    ID = []
    results = []
    num_img = 0
    num_box = 0
    anno = json.load(open(anno_file,'r'))
    for image_info in anno['images']:
        fileName.append(image_info['file_name'])
        ID.append(image_info['id'])
    fileName2ID = dict(zip(fileName,ID))
    file = open(txt_file,'r').readlines()
    for line in file:
        num_img += 1
        result_oneimg = []
        image_ = dict()
        line = line.strip()
        partsOfLine = line.split()
        if line == '\n'  or line.startswith('#'):
            continue
        if partsOfLine[0] == partsOfLine[-1]:
            continue
        image_['name'] = partsOfLine[0].split('/')[-1]
        image_['id'] = fileName2ID[image_['name']]
        boxes = np.array([float(s) for s in partsOfLine[1::]]).reshape(-1,5)
        boxes[:,2] = boxes[:,2] - boxes[:,0] + 1
        boxes[:,3] = boxes[:,3] - boxes[:,1] + 1
        score = boxes[:,-1]
        boxes = boxes[:,:-1]
        for i in range(boxes.shape[0]):
            num_box += 1
            result_ = dict()
            result_['image_id'] = image_['id']
            result_['category_id'] = 1
            result_['bbox'] = boxes[i]
            result_['score'] = score[i]
            results.append(result_)
    return results

def main(anno_file, txt_file, category):
     print anno_file, txt_file
     result = txt2json(anno_file, txt_file) 
     if category == 'person':
         category_id = 1
     elif category == 'head':
         category_id = 2
     elif category == 'face':
         category_id = 3
     else:
         print 'Can only deal with \'person\' \'head\' or \'face\''
     imdb = COCOEval(anno_file, category_id)        
     imdb.evaluate_detections(result = result)

if __name__ == '__main__':
    args = parse_args()
    anno_file = args.anno_file
    txt_file = args.txt_file
    category = args.category
    main(anno_file, txt_file, category)
