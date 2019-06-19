import ipdb
import sys,os,torch,mmcv
from mmcv.runner import load_checkpoint
#下面这句import的时候定位并调用Registry执行了五个模块的注册，详见running解释
from mmdet.models import build_detector	
from mmdet.apis import inference_detector, show_result

if __name__ == '__main__':
	# ipdb.set_trace()
	cfg = mmcv.Config.fromfile('configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py')
	# cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
	cfg.model.pretrained = None		#inference不设置预训练模型
	#inference只传入cfg的model和test配置，其他的都是训练参数
	model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
	_ = load_checkpoint(model, '/mnt/cephfs_wj/cv/wangxu.ailab/ideas_experiments/mmdetection/pretrain_models/fcos_mstrain_640_800_r50_caffe_fpn_gn_2x_4gpu_20190516-f7329d80.pth')
	# _ = load_checkpoint(model, 'weights/latest.pth')

	# print(model)

	# test a single image
	img= mmcv.imread('demo/test.jpg')
	# img= mmcv.imread('/py/mmdetection-master/data/coco/train2014/21.jpg')
	result = inference_detector(model, img, cfg)
	show_result(img, result, show_result = "ret.jpg")

	# # # test a list of folder
	# path='/py/mmdetection/images/'
	# imgs= os.listdir(path)
	# # ipdb.set_trace()
	# for i in range(len(imgs)):
	# 	imgs[i]=os.path.join(path,imgs[i])
	# # imgs = ['/py/pic/4.jpg', '/py/pic/5.jpg']
	# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
	#     print(i, imgs[i])
	#     show_result(imgs[i], result)

