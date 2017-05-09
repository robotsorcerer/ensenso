import json, os, yaml

def get_bounding_boxes():
    faces_dict = dict()
    lefte_dict = dict()
    righte_dict = dict()

    pwd = os.getcwd()
    data_dir = pwd + '/' + '../src/data/'
    with open(data_dir + 'faces_bboxes.txt') as faces_file:
    #     faces = json.load(faces_file)
        faces = yaml.safe_load(faces_file)
        for f in faces['faces']:
            for k, v in f.items():
                faces_dict[k] = v

    #load eye bounding boxes
    with open(data_dir + 'left_eyes_bboxes.txt') as lefte_file:
    #     faces = json.load(faces_file)
        left_eye = yaml.safe_load(lefte_file)
        for f in left_eye['left_eyes']:
            for k, v in f.items():
                lefte_dict[k] = v


    #load right eye bounding boxes
    with open(data_dir + 'right_eyes_bboxes.txt') as righte_file:
    #     faces = json.load(faces_file)
        right_eye = yaml.safe_load(righte_file)
        for f in right_eye['right_eyes']:
            for k, v in f.items():
                righte_dict[k] = v

    faceAndEyesDict = [faces_dict,  lefte_dict,  righte_dict]
    return faceAndEyesDict


"""
    adapted from
    http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
"""

def get_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou
