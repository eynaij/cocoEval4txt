import logging
import copy
from load_roidb import load_roidb, filter_roidb
from det_eval import evaluate_recall, evaluate_ap


def load_coco_test_roidb_eval(config):
    from common.dataset.coco_eval import COCOEval
    #from common.dataset.wider_face_eval import FACE_Eval
    # get roidb
    imglst_path_list = config.dataset.test_imglst_path_list if 'test_imglst_path_list' in config.dataset else None
    seglabellst_path_list = config.dataset.test_seglabellst_path_list if 'test_seglabellst_path_list' in config.dataset else None
    roidb = load_roidb(roidb_path_list=config.dataset.test_roidb_path_list,
                       imglst_path_list=imglst_path_list,
                       seglabellst_path_list=seglabellst_path_list)
    logging.info('total num images for test: {}'.format(len(roidb)))

    roidb, choose_inds = filter_roidb(roidb, config.TEST.filter_strategy, need_inds=True)
    logging.info('total num images for test after sampling: {}'.format(len(roidb)))

    def _load_and_check_coco(anno_path, imageset_index):
        imdb = COCOEval(anno_path)
        imdb.imageset_index = [imdb.imageset_index[i] for i in choose_inds]
        imdb.num_images = len(imdb.imageset_index)
        if imageset_index is None:
            imageset_index = copy.deepcopy(imdb.imageset_index)
        else:
            for i, j in zip(imageset_index, imdb.imageset_index):
                assert i == j
        return imdb, imageset_index
    def _load_and_check_face(anno_path, imageset_index):
        if config.dataset.submit:
            imdb = FACE_Eval(anno_path, submit=True)
        else:
            imdb = FACE_Eval(anno_path)
        imdb.imageset_index = [imdb.imageset_index[i] for i in choose_inds]
        imdb.num_images = len(imdb.imageset_index)
        if imageset_index is None:
            imageset_index = copy.deepcopy(imdb.imageset_index)
        else:
            for i, j in zip(imageset_index, imdb.imageset_index):
                assert i == j
        return imdb, imageset_index
    imdb = None
    imageset_index = None
    #if config.dataset.widerface:
    #    imdb, imageset_index = _load_and_check_face(config.dataset.test_coco_anno_path['det'], imageset_index)
    #else:
    imdb, imageset_index = _load_and_check_coco(config.dataset.test_coco_anno_path['det'], imageset_index)
    seg_imdb = None
    if 'seg' in config.network.task_type:
        seg_imdb, imageset_index = _load_and_check_coco(config.dataset.test_coco_anno_path['stuff'], imageset_index)
    assert imageset_index is not None

    def eval_func(**kwargs):
        task_type = config.network.task_type
        if 'rpn' in task_type and config.TEST.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if config.network.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'rpn_rcnn' in task_type or 'retinanet' in task_type:
            imdb.evaluate_detections(kwargs['all_boxes'], alg=kwargs['alg'] + '-det')
        if 'seg' in task_type:
            seg_imdb.evaluate_stuff(kwargs['all_seg_results'], alg=kwargs['alg'] + '-seg')
        if 'kps' in task_type:
            imdb.evaluate_keypoints(kwargs['all_kps_results'], alg=kwargs['alg'] + '-kps')
        if 'mask' in task_type:
            imdb.evalute_sds(kwargs['all_mask_boxes'], kwargs['all_masks'], alg=kwargs['alg'] + '-mask')
        if 'densepose' in task_type:
            imdb.evalute_densepose(kwargs['all_densepose_boxes'], kwargs['all_densepose'], alg=kwargs['alg'] + '-densepose')
    return roidb, eval_func


def load_hobot_test_roidb_eval(config):
    # get roidb
    roidb = load_roidb(roidb_path_list=config.dataset.test_roidb_path_list,
                       imglst_path_list=config.dataset.test_imglst_path_list,
                       filter_strategy=config.TEST.filter_strategy)
    logging.info('total num images for test: {}'.format(len(roidb)))

    def eval_func(**kwargs):
        if 'rpn' in config.network.task_type and config.TEST.rpn_do_test:
            all_proposals = kwargs['all_proposals']
            for j in range(1, len(all_proposals)):
                logging.info('***************class %d****************' % j)
                gt_class_ind = j if config.network.rpn_rcnn_num_branch > 1 else None
                evaluate_recall(roidb, all_proposals[j], gt_class_ind=gt_class_ind)
        if 'rpn_rcnn' in config.network.task_type:
            evaluate_ap(roidb, kwargs['all_boxes'])

    return roidb, eval_func
