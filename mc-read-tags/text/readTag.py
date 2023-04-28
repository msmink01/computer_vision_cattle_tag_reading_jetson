import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from model import Model
from utils import AttnLabelConverter
from dataset import AlignCollate, PredictedWebcamImage, PredictedBatchOfImages

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Options():
    def __init__(self, model):
        self.workers = 0
        self.batch_size = 192
        self.saved_model = model
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.sensitive = False
        self.PAD = False
        self.Transformation = "TPS"
        self.FeatureExtraction = "ResNet"
        self.SequenceModeling = "BiLSTM"
        self.Prediction = "Attn"
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.num_gpu = torch.cuda.device_count()


def create_digit_read_envir(opt, doPrint=True):
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    if doPrint:
        print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    return converter, model, AlignCollate_demo


def webcam_img(opt, frame, tagsToBeRead, converter, model, AlignCollate_demo, doPrint=True):
    demo_data = PredictedWebcamImage(frame, tagsToBeRead)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    
    # lists to output findings
    img_name_list = []
    predicted_text_list = []
    predicted_confidences_list = []

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)


            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            if doPrint:
                print(f'{dashed_line}\n{head}\n{dashed_line}')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if doPrint:
                    print(type(img_name), type(pred), confidence_score)
                    print(f'{str(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')
                
                img_name_list.append(f'{img_name}')
                predicted_text_list.append(f'{pred}')
                predicted_confidences_list.append(f'{confidence_score:0.4f}')
    

    return img_name_list, predicted_text_list, predicted_confidences_list

def batch_of_images(opt, frameDict, tagsToBeRead, converter, model, AlignCollate_demo, doPrint=True):
    demo_data = PredictedBatchOfImages(frameDict, tagsToBeRead)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()

    # lists to output findings
    img_name_list = []
    predicted_text_list = []
    predicted_confidences_list = []

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)


            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            if doPrint:
                print(f'{dashed_line}\n{head}\n{dashed_line}')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                if doPrint:
                    print(type(img_name), type(pred), confidence_score)
                    print(f'{str(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')

                img_name_list.append(f'{img_name}')
                predicted_text_list.append(f'{pred}')
                predicted_confidences_list.append(f'{confidence_score:0.4f}')


    return img_name_list, predicted_text_list, predicted_confidences_list
