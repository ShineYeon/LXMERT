import json

import numpy as np
import torch
from torch.utils.data import Dataset

from src.param import args
from src.utils import load_obj_tsv

# Load part of the dataset for fast checking.
# 여기 있는 data의 수는 image 수로 대신하기 때문에, 모든 데이터는 이미지와 관련된 것이 사용되어야 함.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class GQADataset:
    """
    json file에 있는 GQA data exmple:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        }
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()

"""
Example in obj tsv:
FILEDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                "attrs_id", "attrs_conf", "numb_boxes", "boxes", "features"]
"""


class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # detection feature를 img_data로 로딩
        # train, valid에 있는 이미지들 모두 Visual Genome에서 왔기 때문에, 메모리 절약 측면에서 이미지 로딩을 버퍼링함
        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:  # 얘네는 항상 testdev에 모든 데이터를 로딩함
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}

        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # 로드된 image feature하고만 data를 저장했음
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # 이미지 정보 가져오기
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # 박스들을 0~1로 normalize
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_h
        boxes[:, (1, 3)] /= img_w
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # target 생성
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        결과를 GQA-challenge에 제출할 수 있는 json file로 dumping함
        GQA json file 제출 요구사항:
            results = [result]
            result = {
                "questionId": str,
                "prediction": str
            }

        :param quesid2ans: question id를 그에 맞는 예측된 답변에 매핑하는 dict
        :param path: json file을 저장할 파일 경로
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)