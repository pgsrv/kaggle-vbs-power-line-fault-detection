import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pretrainedmodels
import tensorboardX as tbx
import torch
import torch.nn.functional as F
import torchvision
from adabound import adabound
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import binarize
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm

from model.custom_nn import BasicCnnAttentionModule, CnnBlockParams, StackAttentionGRU, GruParams, \
    StackAttentionLSTM, StackedGRU, HierarchicalGruParams, HierarchicalAttentionGru, SENetParams, SignalSENet, \
    CnnAttentionMultipleDropoutModule, CnnAttentionMultipleDropoutParams

RANDOM_SEED = 10

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True


class CosineLoss(nn.Module):

    def forward(self, input, target):
        return torch.mean(1 - F.cosine_similarity(input, target))


def cosine_loss(output, target):
    # return torch.ones(output.shape).cuda() - F.cosine_similarity(output, target)
    return


class NnModelWrapper(object, metaclass=ABCMeta):
    def __init__(self, n_class, dropout_rate, save_dir: Path, optimizer_factory=None, loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None,
                 parallel=True):
        self.dropout_rate = dropout_rate
        self.n_class = n_class
        self.create_model()

        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            if torch.cuda.device_count() >= 2:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

        if model_path:
            self.model.load_state_dict(torch.load(str(model_path)))

        self.n_epoch = None
        self._current_epoch = 0
        self._current_max_valid_score = 0
        self._early_stop_count = 0

        self.save_dir = save_dir

        self.create_model_save_path(save_dir)
        self.train_result_path = save_dir.joinpath("result.csv")
        self.train_results = pd.DataFrame()

        self.lr = lr
        if not optimizer_factory:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer_factory(self.model.parameters(), lr)

        self.add_sigmoid = True
        if loss_function == "cosine":
            self.loss_function = CosineLoss()
        elif loss_function == "mae":
            self.loss_function = nn.L1Loss()
        else:
            self.loss_function = nn.BCEWithLogitsLoss()
            self.add_sigmoid = False

        if score_function:
            self.score_function = score_function
        else:
            self.score_function = lambda y_true, y_pred: matthews_corrcoef(y_true.reshape((-1)),
                                                                           binarize(y_pred,
                                                                                    threshold=threshold).reshape((-1)))
        # self.scheduler = StepLR(self.optimizer, step_size=20)

    def create_model_save_path(self, save_dir):
        self.save_path = save_dir.joinpath("model")

    @abstractmethod
    def create_model(self):
        pass

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, n_epochs, train_batch_size,
              valid_batch_size, patience=10, num_workers=0, validation_metric="score"):
        tensorbord_log = self.save_dir.joinpath("tesnsorbord_log")
        tensorbord_log.mkdir(parents=True, exist_ok=True)
        self._tbx_writer = tbx.SummaryWriter(str(tensorbord_log))
        self.clear_history()
        if validation_metric != "score":
            self._current_max_valid_score = - np.inf
        self.patience = patience
        self._train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                            num_workers=num_workers)
        self._valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True,
                                            num_workers=num_workers)
        self.n_epoch = n_epochs

        logger.info("train with data size: {}".format(len(self._train_dataloader.dataset)))
        logger.info("valid with data size: {}".format(len(self._valid_dataloader.dataset)))

        iterator = tqdm(range(n_epochs))
        for epoch in iterator:
            self._current_epoch = epoch + 1
            logger.info("training %d  / %d epochs", self._current_epoch, n_epochs)
            # self.scheduler.step()
            self._train_epoch(epoch)
            self.write_current_result()
            self._valid_epoch(epoch)
            self.write_current_result()

            if validation_metric == "score":
                valid_metric_value = self.train_results["valid_score"][self._current_epoch]
            else:
                valid_metric_value = - self.train_results["valid_loss"][self._current_epoch]

            if valid_metric_value <= self._current_max_valid_score:
                self._early_stop_count += 1
                logger.info("validation metric isn't improved")
            else:
                logger.info("validation metric is improved from %.5f to %.5f",
                            self._current_max_valid_score, valid_metric_value)
                self._current_max_valid_score = valid_metric_value
                self._early_stop_count = 0
                self.save_models()

            if self._early_stop_count >= self.patience:
                logger.info("======early stopped=====")
                self.model.load_state_dict(torch.load(self.save_path))
                iterator.close()
                break

        logger.info("train done! best validation metric : %.5f", self._current_max_valid_score)

        self._tbx_writer.export_scalars_to_json(self.save_dir.joinpath("all_scalars.json"))
        self._tbx_writer.close()
        return self._current_max_valid_score

    def write_current_result(self):
        self.train_results.to_csv(self.train_result_path, encoding="utf-8")

    def clear_history(self):
        self.n_epoch = None
        self._current_epoch = 0
        self.train_results = pd.DataFrame()

        self._current_max_valid_score = 0
        self._early_stop_count = 0

    def _train_epoch(self, epoch):
        self.model.train()

        all_labels = []
        all_outputs = []
        total_loss = 0.0
        for i, data in enumerate(self._train_dataloader):
            inputs = data["image"]
            # print("batch data size {}".format(inputs.size()))
            if torch.cuda.is_available() and not inputs.is_cuda:
                inputs = inputs.cuda()
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            labels = data["label"]
            all_labels.append(labels.cpu().detach().numpy())

            predicted = torch.sigmoid(outputs)

            all_outputs.append(predicted.cpu().detach().numpy())

            labels = labels.to(device)
            if self.add_sigmoid:
                outputs = predicted
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            # self.optimizer.zero_grad()
            total_loss += loss.cpu().detach().item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))
        self.optimizer.zero_grad()

        avg_loss = total_loss / len(self._train_dataloader)
        logger.info("******train loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_results.loc[self._current_epoch, "train_loss"] = avg_loss
        self._tbx_writer.add_scalar('loss/train_loss', avg_loss, epoch)
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        score = self.score_function(all_labels, all_outputs)
        logger.info("******train score at epoch %d: %.5f :" % (self._current_epoch, score))
        self._tbx_writer.add_scalar('score/train_score', score, epoch)
        self.train_results.loc[self._current_epoch, "train_score"] = score

    def _valid_epoch(self, epoch):
        total_loss = 0.0

        all_labels = []
        all_outputs = []
        self.model.eval()
        for i, data in enumerate(self._valid_dataloader):
            inputs = data["image"]

            if torch.cuda.is_available() and not inputs.is_cuda:
                inputs = inputs.cuda()
            outputs = self.model(inputs)

            labels = data["label"]
            all_labels.append(labels.cpu().detach().numpy())

            predicted = torch.sigmoid(outputs)
            all_outputs.append(predicted.cpu().detach().numpy())
            labels = labels.to(device)

            if self.add_sigmoid:
                outputs = predicted
            loss = self.loss_function(outputs, labels)

            total_loss += loss.cpu().detach().item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] validation loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self._valid_dataloader)
        logger.info("******valid loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_results.loc[self._current_epoch, "valid_loss"] = avg_loss
        self._tbx_writer.add_scalar('loss/valid_loss', avg_loss, epoch)
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        score = self.score_function(all_labels, all_outputs)
        self._tbx_writer.add_scalar('score/valid_score', score, epoch)
        logger.info("******valid score at epoch %d: %.5f :" % (self._current_epoch, score))

        self.train_results.loc[self._current_epoch, "valid_score"] = score

    def save_models(self):
        torch.save(self.model.state_dict(), str(self.save_path))
        logger.info("Checkpoint saved")

    def predict(self, dataset: Dataset, batch_size, n_job):
        logger.info("predicting {} samples...".format(len(dataset)))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_job)
        self.model.eval()
        sigmoid = nn.Sigmoid()
        return np.vstack([sigmoid(self.model(x["image"])).cpu().detach().numpy() for x in tqdm(dataloader)])

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()

    @staticmethod
    def create_adabound(parameters, lr):
        return adabound.AdaBound(parameters, lr, final_lr=0.1)


class VGG16Wrapper(NnModelWrapper):

    def create_model(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.required_grad = False
        in_channel = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(in_channel, 32)
        self.model = nn.Sequential(
            vgg16,
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.n_class)
        )


class ResNet50Wrapper(NnModelWrapper):

    def create_model(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.required_grad = False
        out_channel = resnet.fc.out_features
        self.model = nn.Sequential(
            resnet,
            nn.Dropout(self.dropout_rate),
            nn.Linear(out_channel, 32),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.n_class)
        )


class ResNet50ThinFcWrapper(NnModelWrapper):

    def create_model(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.required_grad = False
        in_channel = resnet.fc.in_features
        resnet.fc = nn.Linear(in_channel, 32)
        self.model = nn.Sequential(
            resnet,
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.n_class)
        )


class ResNet18PretrainedWrapper(NnModelWrapper):

    def create_model(self):
        resnet = torchvision.models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.required_grad = False
        in_channel = resnet.fc.in_features
        resnet.fc = nn.Linear(in_channel, 32)
        self.model = nn.Sequential(
            resnet,
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.n_class)
        )


class ResNet10Wrapper(NnModelWrapper):

    def create_model(self):
        resnet = torchvision.models.ResNet(BasicBlock, [1, 1, 1, 1])

        in_channel = resnet.fc.in_features
        resnet.fc = nn.Linear(in_channel, 32)
        self.model = nn.Sequential(
            resnet,
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.n_class)
        )


class SqueezenetWrapper(NnModelWrapper):

    def __init__(self, n_class, save_dir: Path, optimizer_factory=None, loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5):
        super().__init__(n_class, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold)

    def create_model(self):
        squeezenet = torchvision.models.squeezenet1_1(pretrained=True)
        for param in squeezenet.parameters():
            param.required_grad = False
        squeezenet.num_classes = 1
        squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=1)

        self.model = squeezenet


class Densenet121Wrapper(NnModelWrapper):

    def create_model(self):
        densenet = torchvision.models.densenet121(pretrained=True)
        for param in densenet.parameters():
            param.required_grad = False
        in_channel = densenet.classifier.in_features
        densenet.classifier = nn.Linear(in_channel, 1)
        self.model = densenet


class SeNet154Wrapper(NnModelWrapper):

    def create_model(self):
        senet = pretrainedmodels.__dict__["senet154"](num_classes=1000, pretrained='imagenet')
        for param in senet.parameters():
            param.required_grad = False
        senet.dropout = nn.Dropout(self.dropout_rate)
        senet.last_linear = nn.Linear(senet.last_linear.in_features, 1)
        self.model = senet


class XceptionWrapper(NnModelWrapper):

    def create_model(self):
        xception = pretrainedmodels.__dict__["xception"](num_classes=1000, pretrained='imagenet')
        for param in xception.parameters():
            param.required_grad = False

        xception.last_linear = nn.Linear(xception.last_linear.in_features, 32)
        self.model = nn.Sequential(
            xception,
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.n_class)
        )


class ThinCnnWrapper(NnModelWrapper):

    def __init__(self, n_class, num_classes, cnn_blocks: List[CnnBlockParams],
                 last_pool_size, save_dir: Path, optimizer_factory=None, loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None):
        super().__init__(n_class, 0, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path)
        self.num_classes = num_classes
        self.cnn_blocks = cnn_blocks
        self.last_pool_size = last_pool_size

    def create_model(self):
        self.model = BasicCnnAttentionModule(num_classes=self.n_class, cnn_blocks=self.cnn_blocks,
                                             last_pool_size=self.last_pool_size)


class BiGruAttension(NnModelWrapper):

    def __init__(self, model_params: GruParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path)

    def create_model(self):
        self.model = StackAttentionGRU(self.model_params)


class Gru(NnModelWrapper):

    def __init__(self, model_params: GruParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path)

    def create_model(self):
        self.model = StackedGRU(self.model_params)


class BiLstmAttension(NnModelWrapper):

    def __init__(self, model_params: GruParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path)

    def create_model(self):
        self.model = StackAttentionLSTM(self.model_params)


class HierarchicalAttention(NnModelWrapper):

    def __init__(self, model_params: HierarchicalGruParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None,
                 parallel=True):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path, parallel)

    def create_model(self):
        self.model = HierarchicalAttentionGru(self.model_params)


class SENetWrapper(NnModelWrapper):

    def __init__(self, model_params: SENetParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None,
                 parallel=True):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path, parallel)

    def create_model(self):
        self.model = SignalSENet(self.model_params)


class CnnAttentionMultipleDropoutWrapper(NnModelWrapper):

    def __init__(self, model_params: CnnAttentionMultipleDropoutParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None,
                 parallel=True):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path, parallel)

    def create_model(self):
        self.model = CnnAttentionMultipleDropoutModule(self.model_params)


class BasicCnnAttentionWrapper(NnModelWrapper):

    def __init__(self, model_params: CnnAttentionMultipleDropoutParams, save_dir: Path, optimizer_factory=None,
                 loss_function=None,
                 score_function=None, lr=1e-3, weight_decay=1e-4, threshold=0.5, model_path=None,
                 parallel=True):
        self.model_params = model_params
        super().__init__(1, 0.5, save_dir, optimizer_factory, loss_function, score_function, lr,
                         weight_decay, threshold, model_path, parallel)

    def create_model(self):
        self.model = BasicCnnAttentionModule(self.model_params)


class ImagenetTransformers:
    SIZE = 224

    def __init__(self):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transforms = transforms.Compose(transform_list)

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)


class DefaultTransformers:

    def __init__(self, size=224):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ]
        self.transforms = transforms.Compose(transform_list)

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)
