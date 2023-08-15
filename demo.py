import argparse
import numpy as np
import torch

# import datasets
import utils.misc as misc
from utils.box_utils import xywh2xyxy
from utils.visual_bbox import visualBBox
from models import build_model
import datasets.transforms as T
import PIL.Image as Image
import data_loader
from transformers import AutoTokenizer


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # Input config
    # parser.add_argument('--image', type=str, default='xxx', help="input X-ray image.")
    # parser.add_argument('--phrase', type=str, default='xxx', help="input phrase.")
    # parser.add_argument('--bbox', type=str, default='xxx', help="alternative, if you want to show ground-truth bbox")

    # fool
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)
    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG_ca',
                        help="Name of model to be exploited.")


    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true',
                        help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    # parser.add_argument('--data_root', type=str, default='./ln_data/',
    #                     help='path to ReferIt splits data folder')
    # parser.add_argument('--split_root', type=str, default='data',
    #                     help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='MS_CXR', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='demo_cases',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=13, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    # parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    # parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--eval_model', default='released_checkpoint/MedMPG_MS_CXR.pth', type=str)

    # visualization options
    # parser.add_argument('--visualization', action='store_true',
    #                     help="If true, visual the bbox")
    # parser.add_argument('--visual_MHA', action='store_true',
    #                     help="If true, visual the attention maps")

    return parser

def make_transforms(imsize):
    return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 640 # hyper parameters

    ## Some cases from MS_CXR
    # case1
    img_path = "data_demo/649af982-e3af4e3a-75013d30-cdc71514-a34738fd.jpg"
    phrase = 'Small left apical pneumothorax'
    bbox = [332, 28, 141, 48]  # xywh
    # # case2
    # img_path = 'files/p10/p10977201/s59062881/00363400-cee06fa7-8c2ca1f7-2678a170-b3a62a6e.jpg'
    # phrase = 'small apical pneumothorax'
    # bbox = [161, 134, 111, 37]
    # # case3
    # img_path = 'files/p18/p18426683/s59612243/95423e8e-45dff550-563d3eba-b8bc94be-a87f5a1d.jpg'
    # phrase = 'cardiac silhouette enlarged'
    # bbox = [196, 312, 371, 231]
    # # case4
    # img_path = 'files/p10/p10048451/s53489305/4b7f7a4c-18c39245-53724c25-06878595-7e41bb94.jpg'
    # phrase = 'Focal opacity in the lingular lobe'
    # bbox = [467, 373, 131, 189]
    # # case5
    # img_path = 'files/p19/p19757720/s59572378/13255e1f-91b7b172-02baaeee-340ec493-0e531681.jpg'
    # phrase = 'multisegmental right upper lobe consolidation is present'
    # bbox = [9, 86, 232, 278]
    # # case6
    # img_path = 'files/p10/p10469621/s56786891/04e10148-c36f7afb-d0aaf964-152d8a5d-a02ab550.jpg'
    # phrase = 'right middle lobe opacity, suspicious for pneumonia in the proper clinical setting'
    # bbox = [108, 405, 162, 83]
    # # case7
    # img_path = 'files/p10/p10670818/s50191454/1176839d-cf4f677f-d597a1ef-548bc32a-c05429f3.jpg'
    # phrase = 'Newly appeared lingular opacity'
    # bbox = [392, 297, 141, 151]

    bbox = bbox[:2] + [bbox[0]+bbox[2], bbox[1]+bbox[3]] # xywh2xyxy

    ## encode phrase to bert input
    examples = data_loader.read_examples(phrase, 1)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    features = data_loader.convert_examples_to_features(
        examples=examples, seq_length=args.max_query_len, tokenizer=tokenizer, usemarker=None)
    word_id = torch.tensor(features[0].input_ids)  #
    word_mask = torch.tensor(features[0].input_mask)  #

    ## read and transform image
    input_dict = dict()
    img = Image.open(img_path).convert("RGB")
    input_dict['img'] = img
    fake_bbox = torch.tensor(np.array([0,0,0,0], dtype=int)).float() #for avoid bug
    input_dict['box'] = fake_bbox #for avoid bug
    input_dict['text'] = phrase
    transform = make_transforms(imsize=image_size)
    input_dict = transform(input_dict)
    img = input_dict['img']  #
    img_mask = input_dict['mask']  #
    # if bbox is not None:
    #     bbox = input_dict['box']  #

    img_data = misc.NestedTensor(img.unsqueeze(0), img_mask.unsqueeze(0))
    text_data = misc.NestedTensor(word_id.unsqueeze(0), word_mask.unsqueeze(0))

    ## build model
    model = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    ## model infer
    img_data = img_data.to(device)
    text_data = text_data.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_data, text_data)
        pred_box = outputs['pred_box']
        pred_box = xywh2xyxy(pred_box.detach().cpu())*image_size
        pred_box = pred_box.numpy()[0]
        pred_box = [round(pred_box[0]), round(pred_box[1]), round(pred_box[2]), round(pred_box[3])]
        visualBBox(img_path, pred_box, bbox, args.output_dir)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MedMPG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
